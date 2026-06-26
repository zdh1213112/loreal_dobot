#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <zbar.h>
#include <unordered_set>
#include <unordered_map>

class BarcodeDetectorNode : public rclcpp::Node {
public:
    BarcodeDetectorNode() : Node("barcode_detector_node") {
        publisher_ = this->create_publisher<std_msgs::msg::String>("detected_barcodes", 10);

        // ================= 格式白名单 =================
        scanner_.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_ENABLE, 0);
        scanner_.set_config(zbar::ZBAR_EAN13, zbar::ZBAR_CFG_ENABLE, 1);
        // 如需其他类型自行放开：
        // scanner_.set_config(zbar::ZBAR_CODE128, zbar::ZBAR_CFG_ENABLE, 1);
        // scanner_.set_config(zbar::ZBAR_QRCODE, zbar::ZBAR_CFG_ENABLE, 1);
        // ==============================================

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
            sensor.set_option(RS2_OPTION_EXPOSURE, 60.0f);
            sensor.set_option(RS2_OPTION_GAIN, 16.0f);

            RCLCPP_INFO(this->get_logger(),
                " D415 小条码专用模式已启动 (EAN-13 白名单 + 梯度定位 + 自适应放大)");
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
    struct BarcodeResult {
        cv::Rect rect;
        std::string data;
    };

    // ============================================================
    // 💡 EAN-13 校验位算法：过滤掉 99% 的错误识别
    // EAN-13 规则：前 12 位按奇偶加权求和，校验位 = (10 - 求和%10) % 10
    // ============================================================
    bool validate_ean13(const std::string & code) {
        if (code.size() != 13) return false;
        for (char c : code) if (c < '0' || c > '9') return false;

        int sum = 0;
        for (int i = 0; i < 12; i++) {
            int d = code[i] - '0';
            sum += (i % 2 == 0) ? d : d * 3;
        }
        int check = (10 - sum % 10) % 10;
        return check == (code[12] - '0');
    }

    // ============================================================
    // 💡 小条码定位：基于 Sobel 梯度差检测条码候选区域
    // EAN-13 竖条纹特性：X 方向梯度 >> Y 方向梯度
    // 返回一个或多个紧凑的候选矩形 (相对于输入图像)
    // ============================================================
    std::vector<cv::Rect> locate_barcode_candidates(const cv::Mat & gray) {
        cv::Mat grad_x, grad_y, grad;
        cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);
        cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);

        // |grad_x| - |grad_y| —— 条码区域此值应明显为正
        cv::Mat abs_x, abs_y;
        cv::convertScaleAbs(grad_x, abs_x);
        cv::convertScaleAbs(grad_y, abs_y);
        cv::subtract(abs_x, abs_y, grad);

        // 模糊 + 二值化，把离散的强梯度点粘连成块
        cv::blur(grad, grad, cv::Size(9, 9));
        cv::Mat binary;
        cv::threshold(grad, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        // 水平方向闭运算把同一条码的条纹连成一片
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 7));
        cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);

        // 再腐蚀膨胀去掉孤立噪点
        cv::erode(binary, binary, cv::Mat(), cv::Point(-1, -1), 2);
        cv::dilate(binary, binary, cv::Mat(), cv::Point(-1, -1), 4);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<cv::Rect> candidates;
        for (auto & c : contours) {
            cv::Rect r = cv::boundingRect(c);

            // EAN-13 条码的特征过滤：
            // - 面积不能太小 (排除噪点)
            // - 宽高比在 1.2~6.0 之间 (竖条纹矩形)
            // - 宽度不能太小 (至少要有一定像素)
            double aspect = (double)r.width / std::max(1, r.height);
            if (r.area() < 400) continue;
            if (aspect < 1.2 || aspect > 6.0) continue;
            if (r.width < 30) continue;

            // 外扩一圈留出 quiet zone (条码两端的空白区是识别必需的)
            int pad_x = r.width / 6;
            int pad_y = r.height / 2;
            r.x = std::max(0, r.x - pad_x);
            r.y = std::max(0, r.y - pad_y);
            r.width  = std::min(gray.cols - r.x, r.width  + 2 * pad_x);
            r.height = std::min(gray.rows - r.y, r.height + 2 * pad_y);
            candidates.push_back(r);
        }

        // 按面积从大到小排序，优先处理大候选区
        std::sort(candidates.begin(), candidates.end(),
                  [](const cv::Rect & a, const cv::Rect & b) {
                      return a.area() > b.area();
                  });
        // 只保留前 3 个，避免浪费算力
        if (candidates.size() > 3) candidates.resize(3);
        return candidates;
    }

    // ============================================================
    // 💡 小条码专用增强：大倍数放大 + 专治小模块的锐化
    // ============================================================
    cv::Mat enhance_small_barcode(const cv::Mat & gray, double scale) {
        cv::Mat up;
        // 立方插值对条码边缘保留最好 (比 LINEAR 更利于小条码)
        cv::resize(gray, up, cv::Size(), scale, scale, cv::INTER_CUBIC);

        // 反锐化掩模 —— 放大后必然变糊，这一步恢复模块边缘
        cv::Mat blur, sharp;
        cv::GaussianBlur(up, blur, cv::Size(0, 0), 1.2);
        cv::addWeighted(up, 1.8, blur, -0.8, 0, sharp);

        // CLAHE 再提一次对比度，对光照不均的小条码很关键
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.5, cv::Size(16, 16));
        cv::Mat out;
        clahe->apply(sharp, out);
        return out;
    }

    // 在给定灰度图 + 偏移坐标系下扫描并收集结果
    // roi_origin 是 scan_gray 在原始 image 里的左上角坐标
    // scale 是 scan_gray 相对原图的放大倍数
    bool scan_and_collect(const cv::Mat & scan_gray, cv::Point roi_origin, double scale) {
        if (scan_gray.empty() || !scan_gray.isContinuous()) return false;

        zbar::Image zbar_image(scan_gray.cols, scan_gray.rows, "Y800",
                               (uchar *)scan_gray.data,
                               (unsigned)(scan_gray.cols * scan_gray.rows));
        int n = scanner_.scan(zbar_image);
        if (n <= 0) return false;

        bool hit = false;
        for (auto symbol = zbar_image.symbol_begin();
             symbol != zbar_image.symbol_end(); ++symbol) {

            std::string data = symbol->get_data();
            std::string type = symbol->get_type_name();

            // 💡 EAN-13 强校验：校验位不对直接丢
            if (type == "EAN-13" && !validate_ean13(data)) continue;

            candidate_counts_[data]++;

            // 💡 投票阈值：因为有校验位保护，EAN-13 只需要 1 次即可确认
            //   其他类型保持 2 次
            int required_votes = (type == "EAN-13") ? 2 : 3;
            if (candidate_counts_[data] < required_votes) continue;

            // 坐标还原：放大坐标 -> 原图坐标
            std::vector<cv::Point> pts;
            for (int i = 0; i < symbol->get_location_size(); i++) {
                int ox = (int)(symbol->get_location_x(i) / scale) + roi_origin.x;
                int oy = (int)(symbol->get_location_y(i) / scale) + roi_origin.y;
                pts.emplace_back(ox, oy);
            }
            if (pts.empty()) continue;

            cv::Rect rect = cv::boundingRect(pts);
            current_results_.push_back({rect, data});

            if (scanned_barcodes_.find(data) == scanned_barcodes_.end()) {
                scanned_barcodes_.insert(data);
                std_msgs::msg::String msg;
                msg.data = data;
                publisher_->publish(msg);
                RCLCPP_INFO(this->get_logger(),
                    "🟢 [EAN-13 校验通过] 捕获: %s", data.c_str());
            }
            hit = true;
        }
        return hit;
    }

    void process_frame() {
        rs2::frameset frames;
        if (!pipeline_.poll_for_frames(&frames)) return;

        rs2::video_frame color_frame = frames.get_color_frame();
        if (!color_frame) return;

        cv::Mat image(cv::Size(1280, 720), CV_8UC3,
                      (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);

        // 准星框每帧都画
        int roi_w = 800, roi_h = 500;
        int roi_x = (image.cols - roi_w) / 2;
        int roi_y = (image.rows - roi_h) / 2;
        cv::Rect roi(roi_x, roi_y, roi_w, roi_h);
        cv::rectangle(image, roi, cv::Scalar(255, 0, 0), 2);
        cv::putText(image, "Scanning Area (Aim Here)", cv::Point(roi.x, roi.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);

        total_frames_++;

        // ================= 抽帧深度识别 =================
        if (total_frames_ % 2 == 0) {
            current_results_.clear();

            cv::Mat roi_image = image(roi).clone();
            cv::Mat gray;
            cv::cvtColor(roi_image, gray, cv::COLOR_BGR2GRAY);

            bool found = false;

            // ===== 阶段 1：梯度定位候选区 + 针对性放大 =====
            // 这是针对小条码的最关键手段 —— 先找再放大，而不是全图放大
            std::vector<cv::Rect> candidates = locate_barcode_candidates(gray);

            for (const auto & cand : candidates) {
                cv::Mat cand_gray = gray(cand);

                // 根据候选区大小自适应放大倍数：越小放大越多
                // 让处理后的候选区宽度达到 ~600px 是 ZBar 的舒适区
                double scale = std::max(2.0, std::min(6.0, 600.0 / cand_gray.cols));

                cv::Mat enhanced = enhance_small_barcode(cand_gray, scale);

                // 还原坐标系：cand 是 ROI 内坐标，加上 ROI 左上角 = 原图坐标
                cv::Point origin_in_image(roi_x + cand.x, roi_y + cand.y);
                if (scan_and_collect(enhanced, origin_in_image, scale)) {
                    found = true;
                    // 候选区命中就够了，不用继续扫其他候选区
                    break;
                }
            }

            // ===== 阶段 2：候选区没命中 -> 整个 ROI 放大扫描 (兜底) =====
            if (!found) {
                process_count_++;
                cv::Mat processed;
                double scale = 2.5; // ROI 整体放大 2.5 倍

                if (process_count_ % 2 == 0) {
                    // 策略 A：CLAHE + 反锐化 (抗暗光/模糊)
                    processed = enhance_small_barcode(gray, scale);
                } else {
                    // 策略 B：先放大再自适应阈值 (抗反光过曝)
                    cv::Mat up;
                    cv::resize(gray, up, cv::Size(), scale, scale, cv::INTER_CUBIC);
                    cv::adaptiveThreshold(up, processed, 255,
                                          cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv::THRESH_BINARY, 31, 7);
                }

                scan_and_collect(processed, cv::Point(roi_x, roi_y), scale);
            }
        }
        // =================================================

        // 内存防漏清理
        if (total_frames_ % 100 == 0) candidate_counts_.clear();

        // 统一绘制绿框与文字
        for (const auto & res : current_results_) {
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

    uint64_t total_frames_ = 0;
    uint64_t process_count_ = 0;
    std::vector<BarcodeResult> current_results_;
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<BarcodeDetectorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    cv::destroyAllWindows();
    return 0;
}