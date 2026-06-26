#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <zbar.h>
#include <unordered_set>

class BarcodeDetectorNode : public rclcpp::Node {
public:
    BarcodeDetectorNode() : Node("barcode_detector_node") {
        publisher_ = this->create_publisher<std_msgs::msg::String>("detected_barcodes", 10);
        
        // ================= 💡 核心防御 1：格式白名单 =================
        // 先关闭所有类型的条形码检测
        scanner_.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_ENABLE, 0);
        
        // 仅开启你需要检测的类型！(根据你的输出，6902395852674 是 EAN-13 格式)
        scanner_.set_config(zbar::ZBAR_EAN13, zbar::ZBAR_CFG_ENABLE, 1);
        
        // 如果你的场景中还有其他特定条码，可以取消下面的注释单独开启：
        // scanner_.set_config(zbar::ZBAR_CODE128, zbar::ZBAR_CFG_ENABLE, 1);
        // scanner_.set_config(zbar::ZBAR_QRCODE, zbar::ZBAR_CFG_ENABLE, 1);
        // ==============================================================

        scanner_.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_X_DENSITY, 1);
        scanner_.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_Y_DENSITY, 1);

        rs2::config cfg;
        cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, 30);
        
        try {
            auto profile = pipeline_.start(cfg);
            auto sensor = profile.get_device().query_sensors()[1];

            if (sensor.supports(RS2_OPTION_ENABLE_AUTO_EXPOSURE)) {
                sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 0); 
            }
            sensor.set_option(RS2_OPTION_EXPOSURE, 80.0f); 
            // 降低增益，减少过曝泛白
            sensor.set_option(RS2_OPTION_GAIN, 10.0f);

            RCLCPP_INFO(this->get_logger(), " D415 远距离高精度模式已启动！(白名单：默认只有中国商品条码，格式是 EAN-13 + 投票过滤：条码必须在极短时间内连续被检测到 3 次，才被认为是真实有效的条码)");
        } catch (const rs2::error & e) {
            RCLCPP_ERROR(this->get_logger(), "相机启动失败: %s", e.what());
            rclcpp::shutdown();
        }

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(30),
            std::bind(&BarcodeDetectorNode::process_frame, this)
        );
    }

private:
    // 定义一个结构体，用来缓存上一帧检测到的结果，防止跳帧时画面上的绿框闪烁
    struct BarcodeResult {
        cv::Rect rect;
        std::string data;
    };

    void process_frame() {
        rs2::frameset frames;
        // 拿取最新帧，丢弃积压帧
        if (!pipeline_.poll_for_frames(&frames)) return;
        
        rs2::video_frame color_frame = frames.get_color_frame();
        if (!color_frame) return;

        cv::Mat image(cv::Size(1280, 720), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);

        // 无论是否跳帧，准星框每帧都画，保证用户看画面是流畅的
        int roi_w = 800, roi_h = 500;
        int roi_x = (image.cols - roi_w) / 2;
        int roi_y = (image.rows - roi_h) / 2;
        cv::Rect roi(roi_x, roi_y, roi_w, roi_h);
        cv::rectangle(image, roi, cv::Scalar(255, 0, 0), 2);
        cv::putText(image, "Scanning Area (Aim Here)", cv::Point(roi.x, roi.y - 10), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);

        total_frames_++;

        // ================= 💡 核心修改：抽帧识别逻辑 =================
        // 每 3 帧（大约 100 毫秒）才“拍”一次图进行深度识别
        if (total_frames_ % 2 == 0) {
            
            // 清空上一次的显示缓存
            current_results_.clear();

            cv::Mat roi_image = image(roi).clone(); 
            cv::Mat gray, processed;
            cv::cvtColor(roi_image, gray, cv::COLOR_BGR2GRAY);

            double scale = 2.0; //变焦倍数提升到 2.0 倍（专治远距离小条码）
            cv::resize(gray, gray, cv::Size(), scale, scale, cv::INTER_CUBIC);

            // 依然保留轮转策略，但现在它是基于抽帧后的帧数来轮转
            process_count_++;
            if (process_count_ % 2 == 0) {
                // 策略 A：抗暗光/模糊
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
                clahe->apply(gray, processed);
                cv::GaussianBlur(processed, processed, cv::Size(0, 0), 3);
                cv::addWeighted(gray, 1.5, processed, -0.5, 0, processed);
            } else {
                // 策略 B：抗过曝/反光 (自适应阈值 + 腐蚀加粗)
                cv::adaptiveThreshold(gray, processed, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 21, 5);
                // cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
                // cv::erode(processed, processed, element);
            }

            zbar::Image zbar_image(processed.cols, processed.rows, "Y800", (uchar *)processed.data, processed.cols * processed.rows);
            scanner_.scan(zbar_image);

            for (zbar::Image::SymbolIterator symbol = zbar_image.symbol_begin();
                 symbol != zbar_image.symbol_end(); ++symbol) {
                
                std::string barcode_data = symbol->get_data();
                candidate_counts_[barcode_data]++;
                
                if (candidate_counts_[barcode_data] >= 2) { // 抽帧后，投票阈值可以适当降低到 2 次
                    
                    std::vector<cv::Point> pts;
                    for (int i = 0; i < symbol->get_location_size(); i++) {
                        int orig_x = (symbol->get_location_x(i) / scale) + roi_x;
                        int orig_y = (symbol->get_location_y(i) / scale) + roi_y;
                        pts.push_back(cv::Point(orig_x, orig_y));
                    }
                    cv::Rect rect = cv::boundingRect(pts);
                    
                    // 将结果存入缓存
                    current_results_.push_back({rect, barcode_data});

                    if (scanned_barcodes_.find(barcode_data) == scanned_barcodes_.end()) {
                        scanned_barcodes_.insert(barcode_data);
                        auto msg = std_msgs::msg::String();
                        msg.data = barcode_data;
                        publisher_->publish(msg);
                        RCLCPP_INFO(this->get_logger(), "🟢 [极速确认] 捕获条码: %s", barcode_data.c_str());
                    }
                }
            }
        }
        // ==============================================================

        // 内存防漏清理
        if (total_frames_ % 100 == 0) candidate_counts_.clear();

        // 统一绘制绿框与文字：无论这帧有没有跑 ZBar，都把缓存里的框画出来
        for (const auto& res : current_results_) {
            cv::rectangle(image, res.rect, cv::Scalar(0, 255, 0), 3);
            cv::putText(image, res.data, cv::Point(res.rect.x, res.rect.y - 5), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("D415 Real-time High Performance Scanner", image);
        cv::waitKey(1);
    }

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    rs2::pipeline pipeline_;
    zbar::ImageScanner scanner_;
    
    std::unordered_set<std::string> scanned_barcodes_;
    std::unordered_map<std::string, int> candidate_counts_;
    
    // 新增的成员变量
    uint64_t total_frames_ = 0;       // 用于控制跳帧的总帧数
    uint64_t process_count_ = 0;      // 用于控制算法轮转的处理次数
    std::vector<BarcodeResult> current_results_; // 缓存当前识别框
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<BarcodeDetectorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    cv::destroyAllWindows();
    return 0;
}