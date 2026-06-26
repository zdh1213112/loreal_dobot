#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <zbar.h>
#include <unordered_set>

class BarcodeDetectorNode : public rclcpp::Node {
public:
    BarcodeDetectorNode() : Node("barcode_detector_node"), frame_count_(0) {
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
            sensor.set_option(RS2_OPTION_GAIN, 14.0f);

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
    void process_frame() {
        rs2::frameset frames;
        if (!pipeline_.poll_for_frames(&frames)) return;
        
        rs2::video_frame color_frame = frames.get_color_frame();
        if (!color_frame) return;

        cv::Mat image(cv::Size(1280, 720), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);

        int roi_w = 800;
        int roi_h = 500;
        int roi_x = (image.cols - roi_w) / 2;
        int roi_y = (image.rows - roi_h) / 2;
        cv::Rect roi(roi_x, roi_y, roi_w, roi_h);
        
        cv::Mat roi_image = image(roi).clone(); 
        cv::Mat gray, processed;
        cv::cvtColor(roi_image, gray, cv::COLOR_BGR2GRAY);

        double scale = 1.5;
        cv::resize(gray, gray, cv::Size(), scale, scale, cv::INTER_CUBIC);

        frame_count_++;
        if (frame_count_ % 3 == 0) {
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
            clahe->apply(gray, processed);
            cv::GaussianBlur(processed, processed, cv::Size(0, 0), 3);
            cv::addWeighted(gray, 1.5, processed, -0.5, 0, processed);
        } else if (frame_count_ % 3 == 1) {
            // ================= 💡 新增策略 B：抗过曝专精 =================
            // 1. 自适应阈值：无视全局过曝，局部强制黑白二值化
            cv::adaptiveThreshold(gray, processed, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 21, 5);
            
            // 2. 形态学腐蚀：因为过曝导致黑线变细，我们用 2x2 的内核强行将黑线变粗！
            cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
            cv::erode(processed, processed, element);
            // =============================================================
            // cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
            // clahe->apply(gray, processed);
        } else {
            processed = gray;
        }

        zbar::Image zbar_image(processed.cols, processed.rows, "Y800", (uchar *)processed.data, processed.cols * processed.rows);
        scanner_.scan(zbar_image);

        cv::rectangle(image, roi, cv::Scalar(255, 0, 0), 2);
        cv::putText(image, "Scanning Area (Aim Here)", cv::Point(roi.x, roi.y - 10), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);

        for (zbar::Image::SymbolIterator symbol = zbar_image.symbol_begin();
             symbol != zbar_image.symbol_end(); ++symbol) {
            
            std::string barcode_data = symbol->get_data();
            
            // ================= 💡 核心防御 2：多帧投票机制 =================
            // 只有当这个条码累计被识别到 3 次以上时，才确认为真实条码
            candidate_counts_[barcode_data]++;
            
            if (candidate_counts_[barcode_data] >= 3) {
                // 绘制识别框 (确认为真后才画框)
                std::vector<cv::Point> pts;
                for (int i = 0; i < symbol->get_location_size(); i++) {
                    int orig_x = (symbol->get_location_x(i) / scale) + roi_x;
                    int orig_y = (symbol->get_location_y(i) / scale) + roi_y;
                    pts.push_back(cv::Point(orig_x, orig_y));
                }
                cv::Rect rect = cv::boundingRect(pts);
                cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 3);
                cv::putText(image, barcode_data, cv::Point(rect.x, rect.y - 5), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

                if (scanned_barcodes_.find(barcode_data) == scanned_barcodes_.end()) {
                    scanned_barcodes_.insert(barcode_data);
                    auto msg = std_msgs::msg::String();
                    msg.data = barcode_data;
                    publisher_->publish(msg);
                    RCLCPP_INFO(this->get_logger(), "🟢 [已确认] 捕获条码: %s", barcode_data.c_str());
                }
            }
            // ==============================================================
        }

        // 定期清理候选池，防止内存泄漏或幽灵投票累积 (每 100 帧清理一次)
        if (frame_count_ % 100 == 0) {
            candidate_counts_.clear();
        }

        cv::imshow("D415 Long Range Scanner", image);
        cv::waitKey(1);
    }

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    rs2::pipeline pipeline_;
    zbar::ImageScanner scanner_;
    
    // 存放已确认的条码
    std::unordered_set<std::string> scanned_barcodes_;
    // 存放候选条码的投票计数器
    std::unordered_map<std::string, int> candidate_counts_;
    
    uint64_t frame_count_;
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<BarcodeDetectorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    cv::destroyAllWindows();
    return 0;
}