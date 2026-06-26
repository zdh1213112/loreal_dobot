#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <zbar.h>
#include <unordered_set>

class BarcodeDetectorNode : public rclcpp::Node {
public:
    BarcodeDetectorNode() : Node("barcode_detector_node") {
        // 1. 创建 ROS 2 发布者，发布条码数据
        publisher_ = this->create_publisher<std_msgs::msg::String>("detected_barcodes", 10);

        // 2. 初始化 ZBar 扫描器
        scanner_.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_ENABLE, 1);

        // 3. 配置并启动 RealSense D415
        rs2::config cfg;
        cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
        
        try {
            pipeline_.start(cfg);
            RCLCPP_INFO(this->get_logger(), "Intel D415 相机已启动！");
        } catch (const rs2::error & e) {
            RCLCPP_ERROR(this->get_logger(), "RealSense 启动失败: %s", e.what());
            rclcpp::shutdown();
        }

        // 4. 创建定时器，循环处理图像帧 (30Hz 约等于 33ms)
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(33),
            std::bind(&BarcodeDetectorNode::process_frame, this)
        );
    }

private:
    void process_frame() {
        // 等待相机帧
        rs2::frameset frames = pipeline_.wait_for_frames();
        rs2::video_frame color_frame = frames.get_color_frame();
        if (!color_frame) return;

        // 转为 OpenCV Mat
        cv::Mat image(cv::Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        
        // 转灰度图
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        // 将 OpenCV 图像包装为 ZBar 图像
        int width = gray.cols;
        int height = gray.rows;
        uchar *raw = (uchar *)gray.data;
        zbar::Image zbar_image(width, height, "Y800", raw, width * height);

        // 执行扫描
        scanner_.scan(zbar_image);

        // 遍历检测到的条码
        for (zbar::Image::SymbolIterator symbol = zbar_image.symbol_begin();
             symbol != zbar_image.symbol_end(); ++symbol) {
            
            std::string barcode_data = symbol->get_data();
            std::string barcode_type = symbol->get_type_name();

            // 防刷屏去重逻辑
            if (scanned_barcodes_.find(barcode_data) == scanned_barcodes_.end()) {
                scanned_barcodes_.insert(barcode_data);
                RCLCPP_INFO(this->get_logger(), "新条码: %s (类型: %s)", barcode_data.c_str(), barcode_type.c_str());

                // 将数据发布到 ROS 话题
                auto msg = std_msgs::msg::String();
                msg.data = barcode_data;
                publisher_->publish(msg);
            }

            // 绘制边界框
            std::vector<cv::Point> pts;
            for (int i = 0; i < symbol->get_location_size(); i++) {
                pts.push_back(cv::Point(symbol->get_location_x(i), symbol->get_location_y(i)));
            }
            if (pts.size() == 4) {
                for (int i = 0; i < 4; i++) {
                    cv::line(image, pts[i], pts[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
                }
            }

            // 显示文字
            std::string text = barcode_data + " (" + barcode_type + ")";
            cv::putText(image, text, cv::Point(pts[0].x, pts[0].y - 10), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }

        // OpenCV UI 显示 (按 'q' 退出等逻辑通常由 ROS 统一接管，这里仅展示)
        cv::imshow("ROS 2 D415 Barcode Detection", image);
        cv::waitKey(1);
    }

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    rs2::pipeline pipeline_;
    zbar::ImageScanner scanner_;
    std::unordered_set<std::string> scanned_barcodes_;
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<BarcodeDetectorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    cv::destroyAllWindows();
    return 0;
}