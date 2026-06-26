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
        // ==============================================

        scanner_.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_X_DENSITY, 1);
        scanner_.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_Y_DENSITY, 1);

        rs2::config cfg;
        cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, 30);

        try {
            auto profile = pipeline_.start(cfg);
            auto sensor = profile.get_device().query_sensors()[1];

            // ================= 💡 硬件层抗反光 =================
            if (sensor.supports(RS2_OPTION_ENABLE_AUTO_EXPOSURE)) {
                sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 0);
            }
            // 💡 曝光再压低 —— 宁可暗也不能过曝，过曝的条码信息是不可逆丢失
            //   反光场景的黄金原则：保留高光细节优先于保留暗部
            sensor.set_option(RS2_OPTION_EXPOSURE, 50.0f);
            sensor.set_option(RS2_OPTION_GAIN, 16.0f);

            // 开启背光补偿 (如果支持) —— 让相机自己处理高动态场景
            if (sensor.supports(RS2_OPTION_BACKLIGHT_COMPENSATION)) {
                sensor.set_option(RS2_OPTION_BACKLIGHT_COMPENSATION, 1);
            }
            // 关闭白平衡自动调整，防止反光区导致色调抽风
            if (sensor.supports(RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE)) {
                sensor.set_option(RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE, 0);
            }
            // ===================================================

            RCLCPP_INFO(this->get_logger(),
                " D415 抗反光模式启动 (低曝光 + 高光抑制 + 多策略扫描)");
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
    // 💡 反光检测：找出图像中过曝的高光区域
    //    返回 true 表示存在显著反光
    // ============================================================
    bool detect_glare(const cv::Mat & gray, cv::Mat & glare_mask) {
        cv::threshold(gray, glare_mask, 240, 255, cv::THRESH_BINARY);
        int glare_pixels = cv::countNonZero(glare_mask);
        double ratio = (double)glare_pixels / (gray.rows * gray.cols);
        return ratio > 0.01; // 超过 1% 的像素过曝就算有反光
    }

    // ============================================================
    // 💡 反光抑制方法 1：局部亮度归一化 (Retinex 思想简化版)
    //    用大核模糊估计光照，减去后得到"去光照"的反射图
    //    对薄膜反光特别有效 —— 反光本质是光照分量的异常
    // ============================================================
    cv::Mat suppress_glare_retinex(const cv::Mat & gray) {
        cv::Mat gray_f, illum, reflect;
        gray.convertTo(gray_f, CV_32F, 1.0 / 255.0);

        // 大核高斯模糊估计光照分量 (kernel 要足够大，覆盖反光斑)
        cv::GaussianBlur(gray_f, illum, cv::Size(51, 51), 25);

        // log(原图) - log(光照) = log(反射)
        cv::Mat log_gray, log_illum;
        cv::log(gray_f + 0.01f, log_gray);
        cv::log(illum + 0.01f, log_illum);
        reflect = log_gray - log_illum;

        // 归一化到 0~255
        cv::Mat out;
        cv::normalize(reflect, reflect, 0, 255, cv::NORM_MINMAX);
        reflect.convertTo(out, CV_8U);
        return out;
    }

    // ============================================================
    // 💡 反光抑制方法 2：高光 inpaint 修复
    //    把过曝区域 mask 掉，用周边信息填补
    //    代价高但对小面积反光很有效
    // ============================================================
    cv::Mat suppress_glare_inpaint(const cv::Mat & gray, const cv::Mat & glare_mask) {
        cv::Mat mask_dilated;
        cv::dilate(glare_mask, mask_dilated, cv::Mat(), cv::Point(-1, -1), 2);
        cv::Mat out;
        cv::inpaint(gray, mask_dilated, out, 3, cv::INPAINT_TELEA);
        return out;
    }

    // ============================================================
    // 💡 反光抑制方法 3：Top-hat 形态学 (提取暗细节)
    //    反光是亮区，条码的条纹是暗线 —— top-hat 反而能凸显暗条纹
    // ============================================================
    cv::Mat suppress_glare_tophat(const cv::Mat & gray) {
        // black-hat: 闭运算结果 - 原图，专门提取暗细节
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
        cv::Mat blackhat;
        cv::morphologyEx(gray, blackhat, cv::MORPH_BLACKHAT, kernel);
        // 反转：暗细节 -> 亮条纹，白背景
        cv::Mat out;
        cv::threshold(blackhat, out, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
        return out;
    }

    std::vector<cv::Rect> locate_barcode_candidates(const cv::Mat & gray) {
        cv::Mat grad_x, grad_y, grad;
        cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);
        cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);
        cv::Mat abs_x, abs_y;
        cv::convertScaleAbs(grad_x, abs_x);
        cv::convertScaleAbs(grad_y, abs_y);
        cv::subtract(abs_x, abs_y, grad);

        cv::blur(grad, grad, cv::Size(9, 9));
        cv::Mat binary;
        cv::threshold(grad, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 7));
        cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
        cv::erode(binary, binary, cv::Mat(), cv::Point(-1, -1), 2);
        cv::dilate(binary, binary, cv::Mat(), cv::Point(-1, -1), 4);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<cv::Rect> candidates;
        for (auto & c : contours) {
            cv::Rect r = cv::boundingRect(c);
            double aspect = (double)r.width / std::max(1, r.height);
            if (r.area() < 400) continue;
            if (aspect < 1.2 || aspect > 6.0) continue;
            if (r.width < 30) continue;

            int pad_x = r.width / 6;
            int pad_y = r.height / 2;
            r.x = std::max(0, r.x - pad_x);
            r.y = std::max(0, r.y - pad_y);
            r.width  = std::min(gray.cols - r.x, r.width  + 2 * pad_x);
            r.height = std::min(gray.rows - r.y, r.height + 2 * pad_y);
            candidates.push_back(r);
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const cv::Rect & a, const cv::Rect & b) { return a.area() > b.area(); });
        if (candidates.size() > 3) candidates.resize(3);
        return candidates;
    }

    cv::Mat enhance_small_barcode(const cv::Mat & gray, double scale) {
        cv::Mat up;
        cv::resize(gray, up, cv::Size(), scale, scale, cv::INTER_CUBIC);
        cv::Mat blur, sharp;
        cv::GaussianBlur(up, blur, cv::Size(0, 0), 1.2);
        cv::addWeighted(up, 1.8, blur, -0.8, 0, sharp);
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.5, cv::Size(16, 16));
        cv::Mat out;
        clahe->apply(sharp, out);
        return out;
    }

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
            if (type == "EAN-13" && !validate_ean13(data)) continue;

            candidate_counts_[data]++;
            int required_votes = (type == "EAN-13") ? 2 : 3;
            if (candidate_counts_[data] < required_votes) continue;

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

    // ============================================================
    // 💡 核心处理函数：对一个灰度图尝试多种抗反光策略
    //    一旦命中立刻返回，没反光时只跑策略 0 保持高速
    // ============================================================
    bool scan_with_antiglare(const cv::Mat & gray, cv::Point origin, double scale) {
        // 先检测反光
        cv::Mat glare_mask;
        bool has_glare = detect_glare(gray, glare_mask);

        // 策略 0：标准增强 (无反光/轻反光也能过)
        {
            cv::Mat enhanced = enhance_small_barcode(gray, scale);
            if (scan_and_collect(enhanced, origin, scale)) return true;
        }

        // 如果没反光，跑到这里就返回，不浪费时间
        if (!has_glare) return false;

        // ===== 以下只在检测到反光时启用 =====

        // 策略 1：Retinex 去光照 (对大面积反光最有效)
        {
            cv::Mat retinex = suppress_glare_retinex(gray);
            cv::Mat up;
            cv::resize(retinex, up, cv::Size(), scale, scale, cv::INTER_CUBIC);
            if (scan_and_collect(up, origin, scale)) return true;
        }

        // 策略 2：Black-hat 提取暗细节 (对亮背景暗条纹很有效)
        {
            cv::Mat bh = suppress_glare_tophat(gray);
            cv::Mat up;
            cv::resize(bh, up, cv::Size(), scale, scale, cv::INTER_CUBIC);
            if (scan_and_collect(up, origin, scale)) return true;
        }

        // 策略 3：Inpaint 修复 (代价高，放最后)
        //   只在反光面积适中时用 —— 太大修复不了，太小没必要
        double glare_ratio = (double)cv::countNonZero(glare_mask) / (gray.rows * gray.cols);
        if (glare_ratio < 0.15) {
            cv::Mat fixed = suppress_glare_inpaint(gray, glare_mask);
            cv::Mat enhanced = enhance_small_barcode(fixed, scale);
            if (scan_and_collect(enhanced, origin, scale)) return true;
        }

        // 策略 4：强对比度 + 二值化 (有时简单粗暴反而行)
        {
            cv::Mat up;
            cv::resize(gray, up, cv::Size(), scale, scale, cv::INTER_CUBIC);
            cv::Mat bin;
            cv::adaptiveThreshold(up, bin, 255,
                                  cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv::THRESH_BINARY, 31, 10);
            if (scan_and_collect(bin, origin, scale)) return true;
        }

        return false;
    }

    void process_frame() {
        rs2::frameset frames;
        if (!pipeline_.poll_for_frames(&frames)) return;

        rs2::video_frame color_frame = frames.get_color_frame();
        if (!color_frame) return;

        cv::Mat image(cv::Size(1280, 720), CV_8UC3,
                      (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);

        int roi_w = 800, roi_h = 500;
        int roi_x = (image.cols - roi_w) / 2;
        int roi_y = (image.rows - roi_h) / 2;
        cv::Rect roi(roi_x, roi_y, roi_w, roi_h);
        cv::rectangle(image, roi, cv::Scalar(255, 0, 0), 2);
        cv::putText(image, "Scanning Area (Aim Here)", cv::Point(roi.x, roi.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);

        total_frames_++;

        if (total_frames_ % 2 == 0) {
            current_results_.clear();

            cv::Mat roi_image = image(roi).clone();
            cv::Mat gray;
            cv::cvtColor(roi_image, gray, cv::COLOR_BGR2GRAY);

            // 💡 显示反光状态，方便调试
            cv::Mat glare_vis;
            bool roi_has_glare = detect_glare(gray, glare_vis);
            if (roi_has_glare) {
                cv::putText(image, "GLARE DETECTED",
                            cv::Point(roi.x, roi.y + roi.height + 25),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
            }

            bool found = false;

            // ===== 阶段 1：候选区 + 抗反光多策略 =====
            std::vector<cv::Rect> candidates = locate_barcode_candidates(gray);
            for (const auto & cand : candidates) {
                cv::Mat cand_gray = gray(cand);
                double scale = std::max(2.0, std::min(6.0, 600.0 / cand_gray.cols));
                cv::Point origin_in_image(roi_x + cand.x, roi_y + cand.y);
                if (scan_with_antiglare(cand_gray, origin_in_image, scale)) {
                    found = true;
                    break;
                }
            }

            // ===== 阶段 2：候选区没命中 -> 整个 ROI 抗反光扫描 =====
            if (!found) {
                scan_with_antiglare(gray, cv::Point(roi_x, roi_y), 2.5);
            }
        }

        if (total_frames_ % 100 == 0) candidate_counts_.clear();

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