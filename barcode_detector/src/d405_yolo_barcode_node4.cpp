#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>
#include <librealsense2/rs.hpp>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>

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
            RCLCPP_INFO(this->get_logger(), "✅ ONNXRuntime 引擎加载 YOLOv8 成功！");
        } catch (const Ort::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "❌ ONNXRuntime 加载失败: %s", e.what());
            rclcpp::shutdown();
        }

        rs2::config cfg;
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
            RCLCPP_INFO(this->get_logger(), " D405 启动：1280x720 | 交替推理策略 | 平衡速度与精度");
        } catch (const rs2::error & e) {
            RCLCPP_ERROR(this->get_logger(), "相机启动失败: %s", e.what());
            rclcpp::shutdown();
        }

        timer_ = this->create_wall_timer(std::chrono::milliseconds(10), std::bind(&BarcodeDetectorNode::process_frame, this));
        last_time_ = std::chrono::steady_clock::now();
    }

private:
    struct Detection { cv::Rect box; float score; };
    struct HistoryEntry {
        cv::Rect box;
        float    score;
        int      hit_count;
        int      miss_count;
        int      stable_frames;
    };

    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<const char*> input_node_names_ = {"images"};
    std::vector<const char*> output_node_names_ = {"output0"};

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    rs2::pipeline pipeline_;
    
    uint64_t total_frames_ = 0;
    std::chrono::time_point<std::chrono::steady_clock> last_time_;
    double current_fps_ = 0.0;

    std::vector<HistoryEntry> tracked_boxes_;

    // ============== YOLO 参数 ==============
    static constexpr int   YOLO_INPUT     = 640;
    static constexpr int   NUM_ANCHORS    = 8400;

    // 🔥 核心优化：交替推理策略
    // 奇数帧：跑全图（1 次推理，~30ms）
    // 偶数帧：跑 1x2 水平切片（2 次推理，~60ms）
    // 平均 1.5 次/帧 → FPS 提升至 20+
    static constexpr int   SLICE_COLS     = 2;
    static constexpr float SLICE_OVERLAP  = 0.20f;

    static constexpr float CONF_THRESHOLD = 0.40f;
    static constexpr float NMS_THRESHOLD  = 0.50f;
    static constexpr float SCORE_KEEP     = 0.25f;

    static constexpr int   MIN_BOX_W      = 20;
    static constexpr int   MIN_BOX_H      = 10;
    static constexpr float MAX_ASPECT     = 10.0f;

    // 条纹纹理验证
    static constexpr bool  ENABLE_STRIPE_CHECK     = true;
    static constexpr float MIN_STRIPE_MEAN         = 20.0f;
    static constexpr float MIN_STRIPE_STD          = 30.0f;
    static constexpr float MIN_DIRECTIONAL_RATIO   = 1.4f;

    // 时序稳定性
    static constexpr bool  ENABLE_TEMPORAL_FILTER  = true;
    static constexpr int   MIN_HIT_TO_CANDIDATE    = 1;
    static constexpr int   MIN_HIT_TO_CONFIRMED    = 2;   // 🔥 从 3 降到 2：响应更快
    static constexpr int   MAX_MISS_COUNT          = 5;   // 🔥 从 4 增到 5：更宽容
    static constexpr float MATCH_IOU               = 0.25f;

    void draw_modern_ui(cv::Mat& img, const cv::Rect& r, const cv::Scalar& color, const std::string& label, bool is_locked) {
        cv::Rect safe = r & cv::Rect(0, 0, img.cols, img.rows);
        if (safe.width < 4 || safe.height < 4) return;

        int t = is_locked ? 3 : 2; 
        int len = std::max(10, std::min(safe.width, safe.height) / 5); 

        cv::line(img, cv::Point(safe.x, safe.y), cv::Point(safe.x + len, safe.y), color, t);
        cv::line(img, cv::Point(safe.x, safe.y), cv::Point(safe.x, safe.y + len), color, t);
        cv::line(img, cv::Point(safe.x + safe.width, safe.y), cv::Point(safe.x + safe.width - len, safe.y), color, t);
        cv::line(img, cv::Point(safe.x + safe.width, safe.y), cv::Point(safe.x + safe.width, safe.y + len), color, t);
        cv::line(img, cv::Point(safe.x, safe.y + safe.height), cv::Point(safe.x + len, safe.y + safe.height), color, t);
        cv::line(img, cv::Point(safe.x, safe.y + safe.height), cv::Point(safe.x, safe.y + safe.height - len), color, t);
        cv::line(img, cv::Point(safe.x + safe.width, safe.y + safe.height), cv::Point(safe.x + safe.width - len, safe.y + safe.height), color, t);
        cv::line(img, cv::Point(safe.x + safe.width, safe.y + safe.height), cv::Point(safe.x + safe.width, safe.y + safe.height - len), color, t);

        if (is_locked && !label.empty()) {
            cv::Mat overlay; img.copyTo(overlay);
            cv::rectangle(overlay, safe, color, cv::FILLED);
            cv::addWeighted(overlay, 0.15, img, 0.85, 0, img);
            int baseLine; cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseLine);
            int lx = safe.x, ly = std::max(0, safe.y - labelSize.height - 8);
            cv::Rect labelRect(lx, ly, std::min(labelSize.width + 10, img.cols - lx), labelSize.height + 8);
            cv::rectangle(img, labelRect, color, cv::FILLED);
            cv::putText(img, label, cv::Point(safe.x + 5, std::max(labelSize.height + 5, safe.y - 5)), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
        }
    }

    // ================= 条纹纹理验证 =================
    struct StripeMetrics {
        float mean;
        float stddev;
        float directional_ratio;
        bool passes() const {
            return mean >= MIN_STRIPE_MEAN
                && stddev >= MIN_STRIPE_STD
                && directional_ratio >= MIN_DIRECTIONAL_RATIO;
        }
    };

    StripeMetrics compute_stripe_metrics(const cv::Mat& bgr_roi) {
        StripeMetrics m = {0.0f, 0.0f, 0.0f};
        if (bgr_roi.empty() || bgr_roi.cols < 20 || bgr_roi.rows < 10) return m;
        if (bgr_roi.channels() != 3) return m;

        try {
            cv::Mat gray;
            cv::cvtColor(bgr_roi, gray, cv::COLOR_BGR2GRAY);
            if (gray.empty()) return m;

            cv::Mat sobel_x, sobel_y;
            cv::Sobel(gray, sobel_x, CV_32F, 1, 0, 3);
            cv::Sobel(gray, sobel_y, CV_32F, 0, 1, 3);

            cv::Mat abs_x, abs_y;
            cv::convertScaleAbs(sobel_x, abs_x);
            cv::convertScaleAbs(sobel_y, abs_y);

            double mean_x = cv::mean(abs_x)[0];
            double mean_y = cv::mean(abs_y)[0];
            cv::Mat primary  = (mean_x > mean_y) ? abs_x : abs_y;
            double primary_mean = std::max(mean_x, mean_y);
            double secondary_mean = std::min(mean_x, mean_y);

            cv::Scalar std_scalar, mean_scalar;
            cv::meanStdDev(primary, mean_scalar, std_scalar);

            m.mean = (float)primary_mean;
            m.stddev = (float)std_scalar[0];
            m.directional_ratio = secondary_mean > 1e-5 ? (float)(primary_mean / secondary_mean) : 10.0f;
            return m;
        } catch (const cv::Exception&) {
            return m;
        }
    }

    float compute_iou(const cv::Rect& a, const cv::Rect& b) {
        int x1 = std::max(a.x, b.x);
        int y1 = std::max(a.y, b.y);
        int x2 = std::min(a.x + a.width, b.x + b.width);
        int y2 = std::min(a.y + a.height, b.y + b.height);
        int inter_w = std::max(0, x2 - x1);
        int inter_h = std::max(0, y2 - y1);
        int inter = inter_w * inter_h;
        int uni = a.area() + b.area() - inter;
        return uni > 0 ? (float)inter / (float)uni : 0.0f;
    }

    // ================= 时序稳定性 =================
    std::pair<std::vector<Detection>, std::vector<Detection>>
    temporal_filter(const std::vector<Detection>& raw_dets) {
        const size_t orig_size = tracked_boxes_.size();
        std::vector<bool> hist_matched(orig_size, false);
        std::vector<bool> raw_used(raw_dets.size(), false);

        for (size_t i = 0; i < raw_dets.size(); ++i) {
            float best_iou = MATCH_IOU;
            int best_j = -1;
            for (size_t j = 0; j < orig_size; ++j) {
                if (hist_matched[j]) continue;
                float iou = compute_iou(raw_dets[i].box, tracked_boxes_[j].box);
                if (iou > best_iou) { best_iou = iou; best_j = (int)j; }
            }
            if (best_j >= 0) {
                auto& h = tracked_boxes_[best_j];
                float alpha = 0.5f;
                h.box.x      = (int)(alpha * raw_dets[i].box.x      + (1 - alpha) * h.box.x);
                h.box.y      = (int)(alpha * raw_dets[i].box.y      + (1 - alpha) * h.box.y);
                h.box.width  = (int)(alpha * raw_dets[i].box.width  + (1 - alpha) * h.box.width);
                h.box.height = (int)(alpha * raw_dets[i].box.height + (1 - alpha) * h.box.height);
                h.score = std::max(h.score, raw_dets[i].score);
                h.hit_count++;
                h.miss_count = 0;
                h.stable_frames++;
                hist_matched[best_j] = true;
                raw_used[i] = true;
            }
        }

        for (size_t i = 0; i < raw_dets.size(); ++i) {
            if (!raw_used[i]) {
                tracked_boxes_.push_back({raw_dets[i].box, raw_dets[i].score, 1, 0, 1});
            }
        }

        for (size_t j = 0; j < orig_size; ++j) {
            if (!hist_matched[j]) {
                tracked_boxes_[j].miss_count++;
                tracked_boxes_[j].stable_frames = 0;
            }
        }
        tracked_boxes_.erase(
            std::remove_if(tracked_boxes_.begin(), tracked_boxes_.end(),
                [](const HistoryEntry& e) { return e.miss_count > MAX_MISS_COUNT; }),
            tracked_boxes_.end()
        );

        std::vector<Detection> candidates;
        std::vector<Detection> confirmed;
        for (const auto& h : tracked_boxes_) {
            if (h.hit_count >= MIN_HIT_TO_CONFIRMED) {
                confirmed.push_back({h.box, h.score});
            } else if (h.hit_count >= MIN_HIT_TO_CANDIDATE) {
                candidates.push_back({h.box, h.score});
            }
        }
        return {candidates, confirmed};
    }

    inline float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }

    // ================= YOLO 单次推理 =================
    std::vector<Detection> run_yolo_infer(const cv::Mat& tile_bgr) {
        std::vector<Detection> results;
        if (tile_bgr.empty() || tile_bgr.cols < 32 || tile_bgr.rows < 32) return results;

        int src_w = tile_bgr.cols;
        int src_h = tile_bgr.rows;
        float ratio = std::min((float)YOLO_INPUT / src_w, (float)YOLO_INPUT / src_h);
        int new_w = std::round(src_w * ratio);
        int new_h = std::round(src_h * ratio);
        int pad_w = (YOLO_INPUT - new_w) / 2;
        int pad_h = (YOLO_INPUT - new_h) / 2;

        cv::Mat resized;
        cv::resize(tile_bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
        cv::Mat letterbox(YOLO_INPUT, YOLO_INPUT, CV_8UC3, cv::Scalar(114, 114, 114));
        resized.copyTo(letterbox(cv::Rect(pad_w, pad_h, new_w, new_h)));

        cv::Mat blob;
        cv::dnn::blobFromImage(letterbox, blob, 1.0 / 255.0, cv::Size(YOLO_INPUT, YOLO_INPUT), cv::Scalar(), true, false);

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> input_dims = {1, 3, YOLO_INPUT, YOLO_INPUT};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, (float*)blob.data, blob.total(), input_dims.data(), input_dims.size());

        auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
            input_node_names_.data(), &input_tensor, 1,
            output_node_names_.data(), 1);
        float* raw_output = output_tensors[0].GetTensorMutableData<float>();

        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        boxes.reserve(128);
        scores.reserve(128);

        for (int i = 0; i < NUM_ANCHORS; ++i) {
            float s0 = raw_output[4 * NUM_ANCHORS + i];
            float s1 = raw_output[5 * NUM_ANCHORS + i];
            float max_score = std::max(s0, s1);
            if (max_score > 1.0f) max_score = sigmoid(max_score);
            if (max_score < CONF_THRESHOLD) continue;

            float xc = raw_output[0 * NUM_ANCHORS + i];
            float yc = raw_output[1 * NUM_ANCHORS + i];
            float w  = raw_output[2 * NUM_ANCHORS + i];
            float h  = raw_output[3 * NUM_ANCHORS + i];

            float x1f = (xc - 0.5f * w - pad_w) / ratio;
            float y1f = (yc - 0.5f * h - pad_h) / ratio;
            float x2f = (xc + 0.5f * w - pad_w) / ratio;
            float y2f = (yc + 0.5f * h - pad_h) / ratio;

            int left   = std::max(0, (int)std::round(x1f));
            int top    = std::max(0, (int)std::round(y1f));
            int right  = std::min(src_w - 1, (int)std::round(x2f));
            int bottom = std::min(src_h - 1, (int)std::round(y2f));

            int w_real = right - left;
            int h_real = bottom - top;

            if (w_real < MIN_BOX_W || h_real < MIN_BOX_H) continue;
            if (w_real > src_w * 0.92 || h_real > src_h * 0.92) continue;

            float aspect = (float)std::max(w_real, h_real) / (float)std::min(w_real, h_real);
            if (aspect > MAX_ASPECT) continue;

            boxes.emplace_back(left, top, w_real, h_real);
            scores.push_back(max_score);
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, scores, SCORE_KEEP, NMS_THRESHOLD, indices);
        for (int idx : indices) {
            results.push_back({boxes[idx], scores[idx]});
        }
        return results;
    }

    // ================= 1x2 水平切片 =================
    std::vector<Detection> run_sliced_infer(const cv::Mat& image) {
        std::vector<cv::Rect> all_boxes;
        std::vector<float>    all_scores;

        int W = image.cols, H = image.rows;
        int tile_w = W / SLICE_COLS;
        int overlap_w = (int)(tile_w * SLICE_OVERLAP);

        for (int c = 0; c < SLICE_COLS; ++c) {
            int x0 = std::max(0, c * tile_w - overlap_w);
            int x1 = std::min(W, (c + 1) * tile_w + overlap_w);

            cv::Rect tile_rect(x0, 0, x1 - x0, H);
            if (tile_rect.width < 32) continue;
            cv::Mat tile = image(tile_rect);

            auto tile_dets = run_yolo_infer(tile);
            for (auto& d : tile_dets) {
                d.box.x += x0;
                all_boxes.push_back(d.box);
                all_scores.push_back(d.score);
            }
        }

        std::vector<Detection> result;
        if (all_boxes.empty()) return result;
        std::vector<int> indices;
        cv::dnn::NMSBoxes(all_boxes, all_scores, SCORE_KEEP, NMS_THRESHOLD, indices);
        for (int idx : indices) {
            result.push_back({all_boxes[idx], all_scores[idx]});
        }
        return result;
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

        cv::Mat raw_image(cv::Size(1280, 720), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat image = raw_image.clone();

        total_frames_++;

        // 🔥 核心优化：交替推理策略
        // 奇数帧：全图（1 次推理，快，关注大目标）
        // 偶数帧：1x2 切片（2 次推理，关注小目标）
        // 这样平均 1.5 次推理/帧，同时覆盖不同尺寸
        std::vector<Detection> raw_dets;
        const char* mode_tag;
        auto t0 = std::chrono::steady_clock::now();
        if (total_frames_ % 2 == 1) {
            raw_dets = run_yolo_infer(image);
            mode_tag = "FULL";
        } else {
            raw_dets = run_sliced_infer(image);
            mode_tag = "SLICE";
        }
        auto t1 = std::chrono::steady_clock::now();
        double infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // 条纹验证
        std::vector<Detection> texture_passed;
        texture_passed.reserve(raw_dets.size());
        for (const auto& d : raw_dets) {
            if (!ENABLE_STRIPE_CHECK) { texture_passed.push_back(d); continue; }
            cv::Rect safe = d.box & cv::Rect(0, 0, image.cols, image.rows);
            if (safe.width < 20 || safe.height < 10) continue;
            StripeMetrics m = compute_stripe_metrics(image(safe));
            if (!m.passes()) continue;
            texture_passed.push_back(d);
        }

        // 时序过滤（这是关键：即使本帧没推理到的区域，只要历史追踪还在就会继续显示）
        std::vector<Detection> candidates;
        std::vector<Detection> confirmed;
        if (ENABLE_TEMPORAL_FILTER) {
            auto pair = temporal_filter(texture_passed);
            candidates = pair.first;
            confirmed  = pair.second;
        } else {
            confirmed = texture_passed;
        }

        // 绘制
        for (const auto& d : candidates) {
            draw_modern_ui(image, d.box, cv::Scalar(0, 165, 255), "", false);
        }
        for (const auto& d : confirmed) {
            char tag[32];
            snprintf(tag, sizeof(tag), "BARCODE %.2f", d.score);
            draw_modern_ui(image, d.box, cv::Scalar(0, 255, 0), tag, true);
        }

        if (!confirmed.empty()) {
            std_msgs::msg::String msg;
            char buf[128];
            snprintf(buf, sizeof(buf), "confirmed %zu barcodes", confirmed.size());
            msg.data = buf;
            publisher_->publish(msg);
        }

        char hud[200];
        snprintf(hud, sizeof(hud), "FPS: %.1f | %s %.1fms | Raw:%zu Tex:%zu Cand:%zu OK:%zu", 
            current_fps_, mode_tag, infer_ms,
            raw_dets.size(), texture_passed.size(),
            candidates.size(), confirmed.size());
        cv::putText(image, hud, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 165, 255), 2, cv::LINE_AA);

        cv::Mat display;
        cv::resize(image, display, cv::Size(image.cols / 2, image.rows / 2));
        cv::imshow("YOLOv8 Alternating Strategy (Full/Slice)", display);
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