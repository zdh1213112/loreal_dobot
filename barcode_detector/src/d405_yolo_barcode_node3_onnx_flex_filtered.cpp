#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>
#include <librealsense2/rs.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

class BarcodeDetectorNode : public rclcpp::Node {
public:
    BarcodeDetectorNode()
        : Node("barcode_detector_node_onnx_flex_filtered"),
          env_(ORT_LOGGING_LEVEL_WARNING, "YOLO-ORT-FLEX") {

        publisher_ = this->create_publisher<std_msgs::msg::String>("detected_barcodes", 10);

        // std::string model_path = "/home/zdh/ffs_ws/models/best.onnx";
        std::string model_path = "/home/zdh/yolo_one/yolo_train_xense_load_image/outputs/train/obb_demo111/weights/best.onnx";
        // std::string model_path = "/home/zdh/ffs_ws/models/YOLOV8s_Barcode_Detection.onnx";

        try {
            session_options_.SetIntraOpNumThreads(6);
            session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
            initialize_model_io();
            RCLCPP_INFO(this->get_logger(), "ONNXRuntime 引擎加载成功: %s", model_path.c_str());
        } catch (const Ort::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "ONNXRuntime 加载失败: %s", e.what());
            rclcpp::shutdown();
            return;
        }

        rs2::config cfg;
        const std::string camera_sn = "409122274792";
        cfg.enable_device(camera_sn);
        cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, 30);

        try {
            auto profile = pipeline_.start(cfg);
            for (auto&& sensor : profile.get_device().query_sensors()) {
                if (sensor.supports(RS2_OPTION_ENABLE_AUTO_EXPOSURE)) {
                    sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 0.0f);
                }
                if (sensor.supports(RS2_OPTION_EXPOSURE)) {
                    sensor.set_option(RS2_OPTION_EXPOSURE, 8000.0f);
                }
                if (sensor.supports(RS2_OPTION_GAIN)) {
                    sensor.set_option(RS2_OPTION_GAIN, 16.0f);
                }
                if (sensor.supports(RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE)) {
                    sensor.set_option(RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE, 1.0f);
                }
            }
            RCLCPP_INFO(this->get_logger(), "D405 启动：SN=%s, 1280x720 + 1x2 切片（ONNX 自适应 detect 输出）", camera_sn.c_str());
        } catch (const rs2::error& e) {
            RCLCPP_ERROR(this->get_logger(), "相机启动失败: %s", e.what());
            rclcpp::shutdown();
            return;
        }

        timer_ = this->create_wall_timer(std::chrono::milliseconds(10), std::bind(&BarcodeDetectorNode::process_frame, this));
        last_time_ = std::chrono::steady_clock::now();
    }

private:
    struct Detection {
        cv::Rect box;
        float score;
    };

    struct OutputLayout {
        bool valid = false;
        bool channels_first = true;
        bool has_objectness = false;
        bool ambiguous_six_attr = false;
        bool end2end_boxes = false;
        int64_t num_preds = 0;
        int64_t num_attrs = 0;
        int num_classes = 0;
    };

    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_name_storage_;
    std::vector<std::string> output_name_storage_;
    std::vector<const char*> input_node_names_;
    std::vector<const char*> output_node_names_;

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    rs2::pipeline pipeline_;

    uint64_t total_frames_ = 0;
    std::chrono::time_point<std::chrono::steady_clock> last_time_;
    double current_fps_ = 0.0;
    bool logged_runtime_layout_ = false;

    static constexpr int   DEFAULT_YOLO_INPUT = 640;
    static constexpr int   SLICE_ROWS         = 1;
    static constexpr int   SLICE_COLS         = 2;
    static constexpr float SLICE_OVERLAP      = 0.20f;
    static constexpr float CONF_THRESHOLD     = 0.45f;
    static constexpr float NMS_THRESHOLD      = 0.45f;
    static constexpr float SCORE_KEEP         = 0.30f;
    static constexpr float MIN_ASPECT_RATIO   = 1.60f;
    static constexpr float SUSPICIOUS_ASPECT_RATIO = 2.40f;
    static constexpr float SUSPICIOUS_AREA_RATIO   = 0.035f;
    static constexpr float MIN_PATTERN_SCORE       = 0.38f;

    int input_width_ = DEFAULT_YOLO_INPUT;
    int input_height_ = DEFAULT_YOLO_INPUT;

    static std::string shape_to_string(const std::vector<int64_t>& shape) {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) {
                oss << ", ";
            }
            oss << shape[i];
        }
        oss << "]";
        return oss.str();
    }

    static float clamp01(float v) {
        return std::max(0.0f, std::min(1.0f, v));
    }

    float stripe_pattern_score(const cv::Mat& tile_bgr, const cv::Rect& box) const {
        const cv::Rect image_rect(0, 0, tile_bgr.cols, tile_bgr.rows);
        const cv::Rect clipped = box & image_rect;
        if (clipped.width < 18 || clipped.height < 10) {
            return 0.0f;
        }

        cv::Mat gray;
        cv::cvtColor(tile_bgr(clipped), gray, cv::COLOR_BGR2GRAY);

        cv::Mat eq;
        cv::equalizeHist(gray, eq);

        cv::Mat grad_x;
        cv::Mat grad_y;
        cv::Sobel(eq, grad_x, CV_32F, 1, 0, 3);
        cv::Sobel(eq, grad_y, CV_32F, 0, 1, 3);

        const float mean_x = static_cast<float>(cv::mean(cv::abs(grad_x))[0]);
        const float mean_y = static_cast<float>(cv::mean(cv::abs(grad_y))[0]);
        const float dom = std::max(mean_x, mean_y);
        const float ortho = std::min(mean_x, mean_y);
        if (dom <= 1e-3f) {
            return 0.0f;
        }

        const bool vertical_bars = mean_x >= mean_y;
        const float orient_ratio = dom / (ortho + 1e-3f);
        const float orientation_score = clamp01((orient_ratio - 1.05f) / 1.4f);

        cv::Mat reduced;
        cv::reduce(eq, reduced, vertical_bars ? 0 : 1, cv::REDUCE_AVG, CV_32F);
        const int signal_len = vertical_bars ? reduced.cols : reduced.rows;
        if (signal_len < 16) {
            return 0.0f;
        }

        float signal_sum = 0.0f;
        for (int i = 0; i < signal_len; ++i) {
            signal_sum += vertical_bars ? reduced.at<float>(0, i) : reduced.at<float>(i, 0);
        }
        const float signal_mean = signal_sum / static_cast<float>(signal_len);

        int transitions = 0;
        std::vector<int> run_lengths;
        run_lengths.reserve(signal_len / 2);
        int prev_bit = ((vertical_bars ? reduced.at<float>(0, 0) : reduced.at<float>(0, 0)) >= signal_mean) ? 1 : 0;
        int cur_run = 1;
        for (int i = 1; i < signal_len; ++i) {
            const float v = vertical_bars ? reduced.at<float>(0, i) : reduced.at<float>(i, 0);
            const int bit = (v >= signal_mean) ? 1 : 0;
            if (bit == prev_bit) {
                ++cur_run;
            } else {
                run_lengths.push_back(cur_run);
                cur_run = 1;
                prev_bit = bit;
                ++transitions;
            }
        }
        run_lengths.push_back(cur_run);
        if (run_lengths.size() < 6) {
            return 0.0f;
        }

        float run_mean = 0.0f;
        for (int len : run_lengths) {
            run_mean += static_cast<float>(len);
        }
        run_mean /= static_cast<float>(run_lengths.size());
        if (run_mean <= 1e-3f) {
            return 0.0f;
        }

        float run_var = 0.0f;
        for (int len : run_lengths) {
            const float d = static_cast<float>(len) - run_mean;
            run_var += d * d;
        }
        run_var /= static_cast<float>(run_lengths.size());
        const float run_cv = std::sqrt(run_var) / run_mean;

        const float transition_score = clamp01((static_cast<float>(transitions) - 5.0f) / 11.0f);
        const float run_cv_score = clamp01((run_cv - 0.10f) / 0.30f);

        return 0.45f * orientation_score +
               0.30f * transition_score +
               0.25f * run_cv_score;
    }

    void initialize_model_io() {
        Ort::AllocatorWithDefaultOptions allocator;

        const size_t input_count = session_->GetInputCount();
        const size_t output_count = session_->GetOutputCount();
        if (input_count == 0 || output_count == 0) {
            throw std::runtime_error("ONNX 模型没有输入或输出");
        }

        input_name_storage_.clear();
        output_name_storage_.clear();
        input_node_names_.clear();
        output_node_names_.clear();

        input_name_storage_.reserve(input_count);
        output_name_storage_.reserve(output_count);
        input_node_names_.reserve(input_count);
        output_node_names_.reserve(output_count);

        for (size_t i = 0; i < input_count; ++i) {
            auto name = session_->GetInputNameAllocated(i, allocator);
            input_name_storage_.emplace_back(name.get());
        }
        for (size_t i = 0; i < output_count; ++i) {
            auto name = session_->GetOutputNameAllocated(i, allocator);
            output_name_storage_.emplace_back(name.get());
        }
        for (const auto& name : input_name_storage_) {
            input_node_names_.push_back(name.c_str());
        }
        for (const auto& name : output_name_storage_) {
            output_node_names_.push_back(name.c_str());
        }

        const auto input_shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        const auto output_shape = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

        if (input_shape.size() == 4) {
            if (input_shape[2] > 0) {
                input_height_ = static_cast<int>(input_shape[2]);
            }
            if (input_shape[3] > 0) {
                input_width_ = static_cast<int>(input_shape[3]);
            }
        }

        RCLCPP_INFO(this->get_logger(), "ONNX input[0]: name=%s shape=%s",
            input_name_storage_[0].c_str(), shape_to_string(input_shape).c_str());
        RCLCPP_INFO(this->get_logger(), "ONNX output[0]: name=%s shape=%s",
            output_name_storage_[0].c_str(), shape_to_string(output_shape).c_str());
        RCLCPP_INFO(this->get_logger(), "推理输入尺寸采用: %dx%d", input_width_, input_height_);
    }

    int64_t expected_anchor_free_preds() const {
        if (input_width_ <= 0 || input_height_ <= 0) {
            return 0;
        }
        return static_cast<int64_t>(input_width_ / 8) * static_cast<int64_t>(input_height_ / 8) +
               static_cast<int64_t>(input_width_ / 16) * static_cast<int64_t>(input_height_ / 16) +
               static_cast<int64_t>(input_width_ / 32) * static_cast<int64_t>(input_height_ / 32);
    }

    OutputLayout infer_output_layout(const std::vector<int64_t>& shape) const {
        OutputLayout layout;

        if (shape.size() != 2 && shape.size() != 3) {
            return layout;
        }

        int64_t d1 = 0;
        int64_t d2 = 0;
        if (shape.size() == 3) {
            d1 = shape[1];
            d2 = shape[2];
        } else {
            d1 = shape[0];
            d2 = shape[1];
        }

        if (d1 <= 0 || d2 <= 0) {
            return layout;
        }

        const auto looks_like_attr_dim = [](int64_t d) {
            return d >= 5 && d <= 512;
        };

        if (looks_like_attr_dim(d1) && !looks_like_attr_dim(d2)) {
            layout.channels_first = true;
        } else if (!looks_like_attr_dim(d1) && looks_like_attr_dim(d2)) {
            layout.channels_first = false;
        } else {
            layout.channels_first = d1 <= d2;
        }

        layout.num_attrs = layout.channels_first ? d1 : d2;
        layout.num_preds = layout.channels_first ? d2 : d1;

        if (layout.num_attrs < 5 || layout.num_preds <= 0) {
            return OutputLayout{};
        }

        const int64_t af_preds = expected_anchor_free_preds();
        const int64_t ab_preds = af_preds > 0 ? af_preds * 3 : 0;
        const bool matches_anchor_free = (af_preds > 0 && layout.num_preds == af_preds);
        const bool matches_anchor_based = (ab_preds > 0 && layout.num_preds == ab_preds);

        if (!layout.channels_first && layout.num_attrs == 6 && layout.num_preds > 0 && layout.num_preds <= 300) {
            layout.end2end_boxes = true;
            layout.has_objectness = false;
            layout.num_classes = 1;
            layout.valid = true;
            return layout;
        }

        if (layout.num_attrs == 5) {
            layout.has_objectness = false;
            layout.num_classes = 1;
        } else if (layout.num_attrs == 6) {
            layout.ambiguous_six_attr = true;
            if (matches_anchor_based) {
                layout.has_objectness = true;
                layout.num_classes = 1;
            } else {
                layout.has_objectness = false;
                layout.num_classes = 2;
            }
        } else {
            if (matches_anchor_free) {
                layout.has_objectness = false;
            } else if (matches_anchor_based) {
                layout.has_objectness = true;
            } else if (layout.num_attrs >= 32) {
                layout.has_objectness = false;
            } else if ((layout.num_preds % 3) == 0 && layout.num_preds >= 3000) {
                layout.has_objectness = true;
            } else {
                layout.has_objectness = false;
            }
            layout.num_classes = static_cast<int>(layout.num_attrs - (layout.has_objectness ? 5 : 4));
        }

        if (layout.num_classes <= 0) {
            return OutputLayout{};
        }

        layout.valid = true;
        return layout;
    }

    float value_at(const float* raw_output, const OutputLayout& layout, int64_t pred_idx, int attr_idx) const {
        if (layout.channels_first) {
            return raw_output[attr_idx * layout.num_preds + pred_idx];
        }
        return raw_output[pred_idx * layout.num_attrs + attr_idx];
    }

    void log_runtime_layout_once(const std::vector<int64_t>& shape, const OutputLayout& layout) {
        if (logged_runtime_layout_) {
            return;
        }

        if (!layout.valid) {
            RCLCPP_ERROR(this->get_logger(), "无法解析 ONNX 输出 shape=%s，当前代码只支持常见 YOLO detect 输出",
                shape_to_string(shape).c_str());
            logged_runtime_layout_ = true;
            return;
        }

        RCLCPP_INFO(this->get_logger(),
            "运行时输出解析: shape=%s layout=%s preds=%ld attrs=%ld classes=%d score_mode=%s%s%s",
            shape_to_string(shape).c_str(),
            layout.channels_first ? "[1,C,N]" : "[1,N,C]",
            static_cast<long>(layout.num_preds),
            static_cast<long>(layout.num_attrs),
            layout.num_classes,
            layout.end2end_boxes ? "xyxy+score+cls" :
            layout.has_objectness ? "obj*cls" : "max(cls)",
            layout.ambiguous_six_attr ? " (attrs=6，按启发式推断)" : "",
            layout.end2end_boxes ? " (检测到 end2end 导出)" : "");
        logged_runtime_layout_ = true;
    }

    void draw_modern_ui(cv::Mat& img, const cv::Rect& r, const cv::Scalar& color, const std::string& label, bool is_locked) {
        const int t = is_locked ? 3 : 2;
        const int len = std::max(10, std::min(r.width, r.height) / 5);

        cv::line(img, cv::Point(r.x, r.y), cv::Point(r.x + len, r.y), color, t);
        cv::line(img, cv::Point(r.x, r.y), cv::Point(r.x, r.y + len), color, t);
        cv::line(img, cv::Point(r.x + r.width, r.y), cv::Point(r.x + r.width - len, r.y), color, t);
        cv::line(img, cv::Point(r.x + r.width, r.y), cv::Point(r.x + r.width, r.y + len), color, t);
        cv::line(img, cv::Point(r.x, r.y + r.height), cv::Point(r.x + len, r.y + r.height), color, t);
        cv::line(img, cv::Point(r.x, r.y + r.height), cv::Point(r.x, r.y + r.height - len), color, t);
        cv::line(img, cv::Point(r.x + r.width, r.y + r.height), cv::Point(r.x + r.width - len, r.y + r.height), color, t);
        cv::line(img, cv::Point(r.x + r.width, r.y + r.height), cv::Point(r.x + r.width, r.y + r.height - len), color, t);

        if (is_locked) {
            cv::Mat overlay;
            img.copyTo(overlay);
            cv::rectangle(overlay, r, color, cv::FILLED);
            cv::addWeighted(overlay, 0.15, img, 0.85, 0, img);

            int base_line = 0;
            const cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &base_line);
            const int label_y = std::max(0, r.y - label_size.height - 10);
            cv::Rect label_rect(r.x, label_y, label_size.width + 10, label_size.height + 10);
            cv::rectangle(img, label_rect, color, cv::FILLED);
            cv::putText(img, label, cv::Point(r.x + 5, label_y + label_size.height + 1),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
        }
    }

    std::vector<Detection> run_yolo_infer(const cv::Mat& tile_bgr) {
        std::vector<Detection> results;

        const int src_w = tile_bgr.cols;
        const int src_h = tile_bgr.rows;
        const float ratio = std::min(static_cast<float>(input_width_) / static_cast<float>(src_w),
                                     static_cast<float>(input_height_) / static_cast<float>(src_h));
        const int new_w = std::round(src_w * ratio);
        const int new_h = std::round(src_h * ratio);
        const int pad_w = (input_width_ - new_w) / 2;
        const int pad_h = (input_height_ - new_h) / 2;

        cv::Mat resized;
        cv::resize(tile_bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
        cv::Mat letterbox(input_height_, input_width_, CV_8UC3, cv::Scalar(114, 114, 114));
        resized.copyTo(letterbox(cv::Rect(pad_w, pad_h, new_w, new_h)));

        cv::Mat blob;
        cv::dnn::blobFromImage(letterbox, blob, 1.0 / 255.0, cv::Size(input_width_, input_height_), cv::Scalar(), true, false);

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> input_dims = {1, 3, input_height_, input_width_};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, reinterpret_cast<float*>(blob.data), blob.total(), input_dims.data(), input_dims.size());

        auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
            input_node_names_.data(), &input_tensor, 1,
            output_node_names_.data(), 1);

        auto output_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        const auto output_shape = output_info.GetShape();
        const OutputLayout layout = infer_output_layout(output_shape);
        log_runtime_layout_once(output_shape, layout);
        if (!layout.valid) {
            return results;
        }

        const float* raw_output = output_tensors[0].GetTensorData<float>();

        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        boxes.reserve(128);
        scores.reserve(128);

        for (int64_t i = 0; i < layout.num_preds; ++i) {
            const float xc = value_at(raw_output, layout, i, 0);
            const float yc = value_at(raw_output, layout, i, 1);
            const float w  = value_at(raw_output, layout, i, 2);
            const float h  = value_at(raw_output, layout, i, 3);

            float score = 0.0f;
            if (layout.has_objectness) {
                const float obj = value_at(raw_output, layout, i, 4);
                float best_cls = 1.0f;
                if (layout.num_classes > 0) {
                    best_cls = 0.0f;
                    for (int c = 0; c < layout.num_classes; ++c) {
                        best_cls = std::max(best_cls, value_at(raw_output, layout, i, 5 + c));
                    }
                }
                score = obj * best_cls;
            } else {
                for (int c = 0; c < layout.num_classes; ++c) {
                    score = std::max(score, value_at(raw_output, layout, i, 4 + c));
                }
            }

            if (score < CONF_THRESHOLD) {
                continue;
            }

            const float x1f = (xc - 0.5f * w - pad_w) / ratio;
            const float y1f = (yc - 0.5f * h - pad_h) / ratio;
            const float x2f = (xc + 0.5f * w - pad_w) / ratio;
            const float y2f = (yc + 0.5f * h - pad_h) / ratio;

            const int left   = std::max(0, static_cast<int>(std::round(x1f)));
            const int top    = std::max(0, static_cast<int>(std::round(y1f)));
            const int right  = std::min(src_w - 1, static_cast<int>(std::round(x2f)));
            const int bottom = std::min(src_h - 1, static_cast<int>(std::round(y2f)));

            const int w_real = right - left;
            const int h_real = bottom - top;
            if (w_real <= 0 || h_real <= 0) {
                continue;
            }
            if (w_real < 10 || h_real < 7) {
                continue;
            }
            if (w_real > src_w * 0.92 || h_real > src_h * 0.92) {
                continue;
            }

            const float aspect = static_cast<float>(std::max(w_real, h_real)) /
                                 static_cast<float>(std::max(1, std::min(w_real, h_real)));
            if (aspect < MIN_ASPECT_RATIO) {
                continue;
            }
            if (aspect > 12.0f) {
                continue;
            }

            const float area_ratio = static_cast<float>(w_real * h_real) /
                                     static_cast<float>(std::max(1, src_w * src_h));
            if (aspect < SUSPICIOUS_ASPECT_RATIO || area_ratio > SUSPICIOUS_AREA_RATIO) {
                const float pattern_score = stripe_pattern_score(tile_bgr, cv::Rect(left, top, w_real, h_real));
                if (pattern_score < MIN_PATTERN_SCORE) {
                    continue;
                }
            }

            boxes.emplace_back(left, top, w_real, h_real);
            scores.push_back(score);
        }

        if (layout.end2end_boxes) {
            for (int64_t i = 0; i < layout.num_preds; ++i) {
                const float x1_raw = value_at(raw_output, layout, i, 0);
                const float y1_raw = value_at(raw_output, layout, i, 1);
                const float x2_raw = value_at(raw_output, layout, i, 2);
                const float y2_raw = value_at(raw_output, layout, i, 3);
                const float score = value_at(raw_output, layout, i, 4);

                if (score < CONF_THRESHOLD) {
                    continue;
                }

                const float x1f = (x1_raw - pad_w) / ratio;
                const float y1f = (y1_raw - pad_h) / ratio;
                const float x2f = (x2_raw - pad_w) / ratio;
                const float y2f = (y2_raw - pad_h) / ratio;

                const int left   = std::max(0, static_cast<int>(std::round(x1f)));
                const int top    = std::max(0, static_cast<int>(std::round(y1f)));
                const int right  = std::min(src_w - 1, static_cast<int>(std::round(x2f)));
                const int bottom = std::min(src_h - 1, static_cast<int>(std::round(y2f)));

                const int w_real = right - left;
                const int h_real = bottom - top;
                if (w_real <= 0 || h_real <= 0) {
                    continue;
                }
                if (w_real < 10 || h_real < 7) {
                    continue;
                }
                if (w_real > src_w * 0.92 || h_real > src_h * 0.92) {
                    continue;
                }

                const float aspect = static_cast<float>(std::max(w_real, h_real)) /
                                     static_cast<float>(std::max(1, std::min(w_real, h_real)));
                if (aspect < MIN_ASPECT_RATIO) {
                    continue;
                }
                if (aspect > 12.0f) {
                    continue;
                }

                const float area_ratio = static_cast<float>(w_real * h_real) /
                                         static_cast<float>(std::max(1, src_w * src_h));
                if (aspect < SUSPICIOUS_ASPECT_RATIO || area_ratio > SUSPICIOUS_AREA_RATIO) {
                    const float pattern_score = stripe_pattern_score(tile_bgr, cv::Rect(left, top, w_real, h_real));
                    if (pattern_score < MIN_PATTERN_SCORE) {
                        continue;
                    }
                }

                results.push_back({cv::Rect(left, top, w_real, h_real), score});
            }
            return results;
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, scores, SCORE_KEEP, NMS_THRESHOLD, indices);
        for (int idx : indices) {
            results.push_back({boxes[idx], scores[idx]});
        }
        return results;
    }

    std::vector<Detection> run_sliced_infer(const cv::Mat& image) {
        std::vector<Detection> all_dets;
        std::vector<cv::Rect> all_boxes;
        std::vector<float> all_scores;

        const int W = image.cols;
        const int H = image.rows;
        const int tile_w = W / SLICE_COLS;
        const int tile_h = H / SLICE_ROWS;
        const int overlap_w = static_cast<int>(tile_w * SLICE_OVERLAP);
        const int overlap_h = static_cast<int>(tile_h * SLICE_OVERLAP);

        for (int r = 0; r < SLICE_ROWS; ++r) {
            for (int c = 0; c < SLICE_COLS; ++c) {
                const int x0 = std::max(0, c * tile_w - overlap_w);
                const int y0 = std::max(0, r * tile_h - overlap_h);
                const int x1 = std::min(W, (c + 1) * tile_w + overlap_w);
                const int y1 = std::min(H, (r + 1) * tile_h + overlap_h);

                cv::Rect tile_rect(x0, y0, x1 - x0, y1 - y0);
                cv::Mat tile = image(tile_rect);

                auto tile_dets = run_yolo_infer(tile);
                for (auto& d : tile_dets) {
                    d.box.x += x0;
                    d.box.y += y0;
                    all_boxes.push_back(d.box);
                    all_scores.push_back(d.score);
                }
            }
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(all_boxes, all_scores, SCORE_KEEP, NMS_THRESHOLD, indices);
        for (int idx : indices) {
            all_dets.push_back({all_boxes[idx], all_scores[idx]});
        }
        return all_dets;
    }

    void process_frame() {
        rs2::frameset frames;
        try {
            frames = pipeline_.wait_for_frames(100);
        } catch (const rs2::error&) {
            cv::waitKey(1);
            return;
        }

        rs2::video_frame color_frame = frames.get_color_frame();
        if (!color_frame) {
            cv::waitKey(1);
            return;
        }

        const auto now = std::chrono::steady_clock::now();
        const double dt = std::chrono::duration<double>(now - last_time_).count();
        last_time_ = now;
        if (dt > 0.0 && dt < 1.0) {
            if (current_fps_ == 0.0) {
                current_fps_ = 1.0 / dt;
            } else {
                current_fps_ = (current_fps_ * 0.9) + ((1.0 / dt) * 0.1);
            }
        }

        cv::Mat image(cv::Size(1280, 720), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        total_frames_++;

        const auto t0 = std::chrono::steady_clock::now();
        const std::vector<Detection> dets = run_sliced_infer(image);
        const auto t1 = std::chrono::steady_clock::now();
        const double infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        for (const auto& d : dets) {
            char score_text[32];
            snprintf(score_text, sizeof(score_text), "%.2f", d.score);
            draw_modern_ui(image, d.box, cv::Scalar(0, 255, 0), score_text, true);
        }

        if (!dets.empty()) {
            std_msgs::msg::String msg;
            char buf[128];
            snprintf(buf, sizeof(buf), "detected %zu barcodes", dets.size());
            msg.data = buf;
            publisher_->publish(msg);
        }

        char hud_text[160];
        snprintf(hud_text, sizeof(hud_text), "FPS: %.1f | Infer: %.1fms | Dets: %zu | Input: %dx%d",
            current_fps_, infer_ms, dets.size(), input_width_, input_height_);
        cv::putText(image, hud_text, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8,
            cv::Scalar(0, 165, 255), 2, cv::LINE_AA);

        cv::Mat display;
        cv::resize(image, display, cv::Size(image.cols / 2, image.rows / 2));
        cv::imshow("YOLO ONNX Flex Detection Filtered", display);
        cv::waitKey(1);
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<BarcodeDetectorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    cv::destroyAllWindows();
    return 0;
}
