#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>
#include <librealsense2/rs.hpp>
#include <zbar.h>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <chrono>

class BarcodeDetectorNode : public rclcpp::Node {
public:
    BarcodeDetectorNode() : Node("barcode_detector_node"),
        env_(ORT_LOGGING_LEVEL_WARNING, "YOLOv8-ORT") {
        
        publisher_ = this->create_publisher<std_msgs::msg::String>("detected_barcodes", 10);

        std::string model_path = "/home/zdh/ffs_ws/models/YOLOV8s_Barcode_Detection.onnx";
        
        try {
            session_options_.SetIntraOpNumThreads(6); 
            session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
            RCLCPP_INFO(this->get_logger(), "✅ ONNXRuntime 引擎加载成功！(多核加速)");
        } catch (const Ort::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "❌ ONNXRuntime 加载失败: %s", e.what());
            rclcpp::shutdown();
        }

        scanner_.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_ENABLE, 0);
        scanner_.set_config(zbar::ZBAR_EAN13, zbar::ZBAR_CFG_ENABLE, 1);
        scanner_.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_X_DENSITY, 1);
        scanner_.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_Y_DENSITY, 1);

        // ================= 💡 升级 1：恢复 720P，增强远距离小目标检测 =================
        rs2::config cfg;
        cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, 30);
        // ==============================================================================

        try {
            auto profile = pipeline_.start(cfg);
            
            // ================= 💡 升级 2：完美的自动曝光不卡顿策略 =================
            // for (auto&& sensor : profile.get_device().query_sensors()) {
            //     // 1. 开启自动曝光，让相机自己适应不同光线
            //     if (sensor.supports(RS2_OPTION_ENABLE_AUTO_EXPOSURE)) {
            //         sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 0.0f);
            //     }
                
            //     // 2. 💡 解决卡顿的核心魔法：关闭自动曝光优先级！
            //     // 这意味着：宁可画面暗一点，也绝对不允许相机降低 30fps 的出图率去强行曝光！
            //     if (sensor.supports(RS2_OPTION_AUTO_EXPOSURE_PRIORITY)) {
            //         sensor.set_option(RS2_OPTION_AUTO_EXPOSURE_PRIORITY, 0.0f);
            //     }

            //     // 3. 关闭背光补偿，防止反光的塑料膜被强行过曝成白斑
            //     if (sensor.supports(RS2_OPTION_BACKLIGHT_COMPENSATION)) {
            //         sensor.set_option(RS2_OPTION_BACKLIGHT_COMPENSATION, 0.0f);
            //     }

            //     if (sensor.supports(RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE)) {
            //         sensor.set_option(RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE, 1.0f);
            //     }
            // }
            // ==============================================================================
             // ================= 💡 针对 D405：彻底抛弃自动曝光 =================
            for (auto&& sensor : profile.get_device().query_sensors()) {
                // 1. 关掉自动曝光！
                if (sensor.supports(RS2_OPTION_ENABLE_AUTO_EXPOSURE)) {
                    sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 0.0f);
                }
                
                // 2. 💡 D405 专属曝光值 (单位：微秒)
                // 30fps 的物理极限是 33333 微秒。
                // 我们锁死在 8000.0f (即 8 毫秒)！
                // 这个极短的曝光时间不仅能完美压制塑料膜反光，还能强制相机跑满 30fps，绝对不卡顿！
                if (sensor.supports(RS2_OPTION_EXPOSURE)) {
                    sensor.set_option(RS2_OPTION_EXPOSURE, 8000.0f); 
                }

                // 3. 曝光时间变短画面会暗，我们用 32.0f 的高增益把条码暗部提亮
                if (sensor.supports(RS2_OPTION_GAIN)) {
                    sensor.set_option(RS2_OPTION_GAIN, 18.0f);
                }

                // 4. 保持自动白平衡，防止画面颜色发绿或发红
                if (sensor.supports(RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE)) {
                    sensor.set_option(RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE, 1.0f);
                }
            }

            RCLCPP_INFO(this->get_logger(), "🚀 D405 满血版启动：720P + 防掉帧AE + 远距离微距");
        } catch (const rs2::error & e) {
            RCLCPP_ERROR(this->get_logger(), "相机启动失败: %s", e.what());
            rclcpp::shutdown();
        }

        timer_ = this->create_wall_timer(std::chrono::milliseconds(10), std::bind(&BarcodeDetectorNode::process_frame, this));
        last_time_ = std::chrono::steady_clock::now();
    }

private:
    struct BarcodeResult { cv::Rect rect; std::string data; };

    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<const char*> input_node_names_ = {"images"};
    std::vector<const char*> output_node_names_ = {"output0"};

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    rs2::pipeline pipeline_;
    zbar::ImageScanner scanner_;
    
    std::unordered_set<std::string> scanned_barcodes_;
    std::unordered_map<std::string, int> candidate_counts_;
    uint64_t total_frames_ = 0;
    std::vector<BarcodeResult> current_results_;
    
    std::chrono::time_point<std::chrono::steady_clock> last_time_;
    double current_fps_ = 0.0;

    void draw_modern_ui(cv::Mat& img, const cv::Rect& r, const cv::Scalar& color, const std::string& label, bool is_locked) {
        int t = is_locked ? 3 : 2; 
        int len = std::max(10, std::min(r.width, r.height) / 5); 

        cv::line(img, cv::Point(r.x, r.y), cv::Point(r.x + len, r.y), color, t);
        cv::line(img, cv::Point(r.x, r.y), cv::Point(r.x, r.y + len), color, t);
        cv::line(img, cv::Point(r.x + r.width, r.y), cv::Point(r.x + r.width - len, r.y), color, t);
        cv::line(img, cv::Point(r.x + r.width, r.y), cv::Point(r.x + r.width, r.y + len), color, t);
        cv::line(img, cv::Point(r.x, r.y + r.height), cv::Point(r.x + len, r.y + r.height), color, t);
        cv::line(img, cv::Point(r.x, r.y + r.height), cv::Point(r.x, r.y + r.height - len), color, t);
        cv::line(img, cv::Point(r.x + r.width, r.y + r.height), cv::Point(r.x + r.width - len, r.y + r.height), color, t);
        cv::line(img, cv::Point(r.x + r.width, r.y + r.height), cv::Point(r.x + r.width, r.y + r.height - len), color, t);

        if (is_locked) {
            cv::Mat overlay; img.copyTo(overlay);
            cv::rectangle(overlay, r, color, cv::FILLED);
            cv::addWeighted(overlay, 0.15, img, 0.85, 0, img);
            int baseLine; cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseLine);
            cv::Rect labelRect(r.x, r.y - labelSize.height - 10, labelSize.width + 10, labelSize.height + 10);
            cv::rectangle(img, labelRect, color, cv::FILLED);
            cv::putText(img, label, cv::Point(r.x + 5, r.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
        }
    }

    bool validate_ean13(const std::string & code) {
        if (code.size() != 13) return false;
        for (char c : code) if (c < '0' || c > '9') return false;
        int sum = 0;
        for (int i = 0; i < 12; i++) { int d = code[i] - '0'; sum += (i % 2 == 0) ? d : d * 3; }
        return ((10 - sum % 10) % 10) == (code[12] - '0');
    }

    cv::Mat rotate_image_safely(const cv::Mat& src, double angle) {
        if (angle == 0.0) return src.clone();
        cv::Point2f center(src.cols / 2.0f, src.rows / 2.0f);
        cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
        cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), src.size(), angle).boundingRect2f();
        rot.at<double>(0, 2) += bbox.width / 2.0 - center.x; rot.at<double>(1, 2) += bbox.height / 2.0 - center.y;
        cv::Mat dst; cv::warpAffine(src, dst, rot, bbox.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255));
        return dst;
    }

    bool detect_glare(const cv::Mat & gray, cv::Mat & glare_mask) {
        cv::threshold(gray, glare_mask, 240, 255, cv::THRESH_BINARY);
        return ((double)cv::countNonZero(glare_mask) / (gray.rows * gray.cols)) > 0.01;
    }

    cv::Mat suppress_glare_retinex(const cv::Mat & gray) {
        cv::Mat gray_f, illum, reflect; gray.convertTo(gray_f, CV_32F, 1.0 / 255.0);
        cv::GaussianBlur(gray_f, illum, cv::Size(51, 51), 25);
        cv::log(gray_f + 0.01f, gray_f); cv::log(illum + 0.01f, illum);
        reflect = gray_f - illum;
        cv::Mat out; cv::normalize(reflect, reflect, 0, 255, cv::NORM_MINMAX);
        reflect.convertTo(out, CV_8U); return out;
    }

    cv::Mat suppress_glare_inpaint(const cv::Mat & gray, const cv::Mat & glare_mask) {
        cv::Mat mask_dilated, out; cv::dilate(glare_mask, mask_dilated, cv::Mat(), cv::Point(-1, -1), 2);
        cv::inpaint(gray, mask_dilated, out, 3, cv::INPAINT_TELEA); return out;
    }

    cv::Mat suppress_glare_tophat(const cv::Mat & gray) {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
        cv::Mat blackhat, out; cv::morphologyEx(gray, blackhat, cv::MORPH_BLACKHAT, kernel);
        cv::threshold(blackhat, out, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU); return out;
    }

    cv::Mat enhance_small_barcode(const cv::Mat & gray, double scale) {
        cv::Mat up, blur, sharp, out; cv::resize(gray, up, cv::Size(), scale, scale, cv::INTER_CUBIC);
        cv::GaussianBlur(up, blur, cv::Size(0, 0), 1.2); cv::addWeighted(up, 1.8, blur, -0.8, 0, sharp); 
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.5, cv::Size(16, 16)); clahe->apply(sharp, out); return out;
    }

    std::string decode_roi(const cv::Mat & scan_gray) {
        if (scan_gray.empty() || !scan_gray.isContinuous()) return "";
        zbar::Image zbar_image(scan_gray.cols, scan_gray.rows, "Y800", (uchar *)scan_gray.data, scan_gray.cols * scan_gray.rows);
        if (scanner_.scan(zbar_image) <= 0) return "";
        for (auto symbol = zbar_image.symbol_begin(); symbol != zbar_image.symbol_end(); ++symbol) {
            std::string data = symbol->get_data(); if (validate_ean13(data)) return data; 
        }
        return "";
    }

    std::string scan_with_antiglare(const cv::Mat & roi_gray, double scale) {
        cv::Mat glare_mask; bool has_glare = detect_glare(roi_gray, glare_mask);
        std::string data = decode_roi(enhance_small_barcode(roi_gray, scale)); if (!data.empty()) return data;
        if (!has_glare) return "";
        cv::Mat up1; cv::resize(suppress_glare_retinex(roi_gray), up1, cv::Size(), scale, scale, cv::INTER_CUBIC);
        if (!(data = decode_roi(up1)).empty()) return data;
        cv::Mat up2; cv::resize(suppress_glare_tophat(roi_gray), up2, cv::Size(), scale, scale, cv::INTER_CUBIC);
        if (!(data = decode_roi(up2)).empty()) return data;
        if (((double)cv::countNonZero(glare_mask) / (roi_gray.rows * roi_gray.cols)) < 0.15) {
            if (!(data = decode_roi(enhance_small_barcode(suppress_glare_inpaint(roi_gray, glare_mask), scale))).empty()) return data;
        }
        cv::Mat up3, bin; cv::resize(roi_gray, up3, cv::Size(), scale, scale, cv::INTER_CUBIC);
        cv::adaptiveThreshold(up3, bin, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 31, 10);
        return decode_roi(bin);
    }

    void process_frame() {
        rs2::frameset frames;
        try { frames = pipeline_.wait_for_frames(100); } catch (const rs2::error& e) { cv::waitKey(1); return; }
        rs2::video_frame color_frame = frames.get_color_frame();
        if (!color_frame) { cv::waitKey(1); return; }

        auto now = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(now - last_time_).count();
        last_time_ = now;
        if (dt > 0.0 && dt < 1.0) {
            if (current_fps_ == 0.0) current_fps_ = 1.0 / dt;
            else current_fps_ = (current_fps_ * 0.9) + ((1.0 / dt) * 0.1);
        }

        // 💡 恢复 1280x720 原始矩阵
        cv::Mat image(cv::Size(1280, 720), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        total_frames_++;

        if (total_frames_ % 2 == 0) {
            current_results_.clear();

            float scale_ratio = std::min(640.0f / image.cols, 640.0f / image.rows);
            int new_w = std::round(image.cols * scale_ratio);
            int new_h = std::round(image.rows * scale_ratio);
            
            cv::Mat resized; cv::resize(image, resized, cv::Size(new_w, new_h));
            cv::Mat letterbox = cv::Mat::zeros(640, 640, CV_8UC3); 
            int pad_w = (640 - new_w) / 2; int pad_h = (640 - new_h) / 2;
            resized.copyTo(letterbox(cv::Rect(pad_w, pad_h, new_w, new_h)));

            cv::Mat blob; cv::dnn::blobFromImage(letterbox, blob, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);

            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            std::vector<int64_t> input_dims = {1, 3, 640, 640};
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)blob.data, blob.total(), input_dims.data(), input_dims.size());

            auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_node_names_.data(), &input_tensor, 1, output_node_names_.data(), 1);
            float* raw_output = output_tensors[0].GetTensorMutableData<float>();
            
            std::vector<cv::Rect> boxes;
            std::vector<float> scores;

            for (int i = 0; i < 8400; ++i) {
                float max_score = std::max(raw_output[4 * 8400 + i], raw_output[5 * 8400 + i]);

                // 💡 升级 3：防止巨型框铺满屏幕。稍微提高阈值到 0.25
                if (max_score > 0.45) { 
                    float xc = raw_output[0 * 8400 + i];
                    float yc = raw_output[1 * 8400 + i];
                    float w  = raw_output[2 * 8400 + i];
                    float h  = raw_output[3 * 8400 + i];

                    // 严密限制边界框不能越界
                    int left = std::max(0, int((xc - 0.5 * w - pad_w) / scale_ratio));
                    int top = std::max(0, int((yc - 0.5 * h - pad_h) / scale_ratio));
                    int right = std::min(image.cols - 1, int((xc + 0.5 * w - pad_w) / scale_ratio));
                    int bottom = std::min(image.rows - 1, int((yc + 0.5 * h - pad_h) / scale_ratio));

                    int width = right - left;
                    int height = bottom - top;

                    // 💡 物理尺寸拦截器：条码不可能占据整个 720P 屏幕的 80% 以上！
                    // 直接干掉那些荒谬的、铺满全屏的异常锚框
                    if (width > 0 && height > 0 && width < image.cols * 0.8 && height < image.rows * 0.8) {
                        boxes.push_back(cv::Rect(left, top, width, height));
                        scores.push_back(max_score);
                    }
                }
            }

            std::vector<int> indices;
            cv::dnn::NMSBoxes(boxes, scores, 0.25, 0.4, indices);

            for (int idx : indices) {
                cv::Rect r = boxes[idx];

                // 💡 升级 4：重写防越界裁剪逻辑，彻底解决铺满屏幕边缘的问题
                int pad = 20;
                int x1 = std::max(0, r.x - pad);
                int y1 = std::max(0, r.y - pad);
                int x2 = std::min(image.cols - 1, r.x + r.width + pad);
                int y2 = std::min(image.rows - 1, r.y + r.height + pad);
                
                r.x = x1;
                r.y = y1;
                r.width = x2 - x1;
                r.height = y2 - y1;
                
                cv::Mat roi_gray;
                cv::cvtColor(image(r), roi_gray, cv::COLOR_BGR2GRAY);
                
                draw_modern_ui(image, r, cv::Scalar(255, 150, 0), "", false);

                std::vector<double> angles = {0, 30, -30, 60, -60, 90};
                for (double angle : angles) {
                    cv::Mat rotated_roi = rotate_image_safely(roi_gray, angle);
                    
                    // 💡 升级 5：远距离微距放大。强制将截取的小目标放大最少 4 倍，保证 ZBar 能看清线条
                    double scale = std::max(4.0, std::min(8.0, 600.0 / rotated_roi.cols));
                    
                    std::string data = scan_with_antiglare(rotated_roi, scale);

                    if (!data.empty()) {
                        candidate_counts_[data]++;
                        if (candidate_counts_[data] >= 2) { 
                            current_results_.push_back({r, data});
                            if (scanned_barcodes_.find(data) == scanned_barcodes_.end()) {
                                scanned_barcodes_.insert(data);
                                std_msgs::msg::String msg; msg.data = data; publisher_->publish(msg);
                                RCLCPP_INFO(this->get_logger(), "🟢 完美捕获: %s (ORT推理, %.0f° 纠偏成功)", data.c_str(), angle);
                            }
                        }
                        break; 
                    }
                }
            }
        }

        if (total_frames_ % 100 == 0) candidate_counts_.clear();
        
        for (const auto & res : current_results_) {
            draw_modern_ui(image, res.rect, cv::Scalar(0, 255, 0), res.data, true);
        }
        
        char fps_text[32];
        snprintf(fps_text, sizeof(fps_text), "FPS: %.1f", current_fps_);
        cv::putText(image, fps_text, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 165, 255), 2, cv::LINE_AA);

        cv::imshow("YOLOv8 + ORT C++ Cybernetics", image);
        cv::waitKey(1); 
    }
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<BarcodeDetectorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    cv::destroyAllWindows();
    return 0;
}