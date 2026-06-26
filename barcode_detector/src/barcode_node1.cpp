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
        
        // 1. ZBar 配置优化：保持最高密度扫描
        scanner_.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_ENABLE, 1);
        scanner_.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_X_DENSITY, 1);
        scanner_.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_Y_DENSITY, 1);

        // 2. RealSense 配置与底层参数调优
        rs2::config cfg;
        // 尝试提高到 60fps 以减少物理帧间隔带来的运动模糊 (D415 支持 640x480 @ 60fps)
        cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
        
        try {
            auto profile = pipeline_.start(cfg);
            auto sensor = profile.get_device().query_sensors()[1]; // 获取彩色传感器

            // --- 💡 硬件优化：解决运动模糊的关键 ---
            if (sensor.supports(RS2_OPTION_ENABLE_AUTO_EXPOSURE)) {
                sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 0); // 关闭自动曝光
            }
            // 设置较短的曝光时间（单位通常是微秒）。对于移动物体，建议 100-300。
            // 值越小，拖影越少，但画面越暗。如果太暗请配合环境光补光。
            sensor.set_option(RS2_OPTION_EXPOSURE, 80.0f); 
            // 补偿增益，提高画面亮度（但会增加噪点，CLAHE 可以过滤）
            sensor.set_option(RS2_OPTION_GAIN, 24.0f);

            RCLCPP_INFO(this->get_logger(), "D415 动态增强模式已启动：低曝光、高帧率控制。");
        } catch (const rs2::error & e) {
            RCLCPP_ERROR(this->get_logger(), "相机启动失败: %s", e.what());
            rclcpp::shutdown();
        }

        // 使用更短的定时器间隔 (15ms 对应约 60fps)
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(15),
            std::bind(&BarcodeDetectorNode::process_frame, this)
        );
    }

private:
    void process_frame() {
        rs2::frameset frames;
        // 使用非阻塞获取，确保实时性
        if (!pipeline_.poll_for_frames(&frames)) return;
        
        rs2::video_frame color_frame = frames.get_color_frame();
        if (!color_frame) return;

        cv::Mat image(cv::Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat gray, processed;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        // --- 💡 算法优化：多级增强策略 ---
        frame_count_++;
        
        if (frame_count_ % 3 == 0) {
            // 策略 A (每3帧一次)：重度增强 - 应对极端模糊
            // 1. CLAHE 局部对比度增强
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
            clahe->apply(gray, processed);
            // 2. 锐化：Unsharp Mask 提升条码边缘锐度
            cv::GaussianBlur(processed, processed, cv::Size(0, 0), 3);
            cv::addWeighted(gray, 1.5, processed, -0.5, 0, processed);
        } 
        else if (frame_count_ % 3 == 1) {
            // 策略 B (每3帧一次)：中度增强 - 应对中等光照
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
            clahe->apply(gray, processed);
        }
        else {
            // 策略 C (其余帧)：原始灰度 - 保持最高处理速度
            processed = gray;
        }

        // 3. 执行解码
        zbar::Image zbar_image(processed.cols, processed.rows, "Y800", (uchar *)processed.data, processed.cols * processed.rows);
        scanner_.scan(zbar_image);

        // 4. 解析与反馈
        bool found = false;
        for (zbar::Image::SymbolIterator symbol = zbar_image.symbol_begin();
             symbol != zbar_image.symbol_end(); ++symbol) {
            
            found = true;
            std::string barcode_data = symbol->get_data();
            
            // 绘制逻辑
            std::vector<cv::Point> pts;
            for (int i = 0; i < symbol->get_location_size(); i++) 
                pts.push_back(cv::Point(symbol->get_location_x(i), symbol->get_location_y(i)));
            
            cv::Rect rect = cv::boundingRect(pts);
            cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2);
            cv::putText(image, symbol->get_type_name(), cv::Point(rect.x, rect.y - 5), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

            // 实时发布与打印
            if (scanned_barcodes_.find(barcode_data) == scanned_barcodes_.end()) {
                scanned_barcodes_.insert(barcode_data);
                auto msg = std_msgs::msg::String();
                msg.data = barcode_data;
                publisher_->publish(msg);
                RCLCPP_INFO(this->get_logger(), "🟢 捕获动态条码: %s", barcode_data.c_str());
            }
        }

        // 5. 调试显示 (显示当前处理过的图像，便于观察增强效果)
        cv::imshow("Processing Pipeline", processed); 
        cv::imshow("Real-time Result", image);
        cv::waitKey(1);
    }

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    rs2::pipeline pipeline_;
    zbar::ImageScanner scanner_;
    std::unordered_set<std::string> scanned_barcodes_;
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