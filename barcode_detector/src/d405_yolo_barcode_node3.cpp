#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>
#include <librealsense2/rs.hpp>
#include <vector>
#include <chrono>

class BarcodeDetectorNode : public rclcpp::Node {
public:
    BarcodeDetectorNode() : Node("barcode_detector_node"),
        env_(ORT_LOGGING_LEVEL_WARNING, "YOLOv8-ORT") {
        
        publisher_ = this->create_publisher<std_msgs::msg::String>("detected_barcodes", 10);

        // std::string model_path = "/home/zdh/ffs_ws/models/YOLOV8s_Barcode_Detection.onnx";
        std::string model_path = "/home/zdh/ffs_ws/models/best.onnx";
        
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
            RCLCPP_INFO(this->get_logger(), " D405 启动：1280x720 + 1x2 切片（纯 YOLO 检测，同步模式）");
        } catch (const rs2::error & e) {
            RCLCPP_ERROR(this->get_logger(), "相机启动失败: %s", e.what());
            rclcpp::shutdown();
        }

        timer_ = this->create_wall_timer(std::chrono::milliseconds(10), std::bind(&BarcodeDetectorNode::process_frame, this));
        last_time_ = std::chrono::steady_clock::now();
    }

private:
    struct Detection { cv::Rect box; float score; };

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

    // ============== YOLO 参数 ==============
    static constexpr int   YOLO_INPUT     = 640;
    static constexpr int   NUM_ANCHORS    = 8400;

    // 🔥 1x2 水平切片：2 次推理，不跑全图兜底（避免冗余）
    static constexpr int   SLICE_ROWS     = 1;
    static constexpr int   SLICE_COLS     = 2;
    static constexpr float SLICE_OVERLAP  = 0.20f;  // 20% 重叠防切断

    static constexpr float CONF_THRESHOLD = 0.45f;
    static constexpr float NMS_THRESHOLD  = 0.45f;
    static constexpr float SCORE_KEEP     = 0.30f;

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

    // ================= YOLO 单次推理 =================
    std::vector<Detection> run_yolo_infer(const cv::Mat& tile_bgr) {
        std::vector<Detection> results;

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

            if (w_real < 10 || h_real < 7) continue;
            if (w_real > src_w * 0.92 || h_real > src_h * 0.92) continue;

            float aspect = (float)std::max(w_real, h_real) / (float)std::min(w_real, h_real);
            if (aspect > 12.0f) continue;

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

    // ================= 切片推理（1x2 水平）=================
    std::vector<Detection> run_sliced_infer(const cv::Mat& image) {
        std::vector<Detection> all_dets;
        std::vector<cv::Rect> all_boxes;
        std::vector<float>    all_scores;

        int W = image.cols, H = image.rows;
        int tile_w = W / SLICE_COLS;
        int tile_h = H / SLICE_ROWS;
        int overlap_w = (int)(tile_w * SLICE_OVERLAP);
        int overlap_h = (int)(tile_h * SLICE_OVERLAP);

        for (int r = 0; r < SLICE_ROWS; ++r) {
            for (int c = 0; c < SLICE_COLS; ++c) {
                int x0 = std::max(0, c * tile_w - overlap_w);
                int y0 = std::max(0, r * tile_h - overlap_h);
                int x1 = std::min(W, (c + 1) * tile_w + overlap_w);
                int y1 = std::min(H, (r + 1) * tile_h + overlap_h);

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

        // 全局 NMS 去重
        std::vector<int> indices;
        cv::dnn::NMSBoxes(all_boxes, all_scores, SCORE_KEEP, NMS_THRESHOLD, indices);
        for (int idx : indices) {
            all_dets.push_back({all_boxes[idx], all_scores[idx]});
        }
        return all_dets;
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

        cv::Mat image(cv::Size(1280, 720), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        total_frames_++;

        // 🔥 同步推理：检测框完美贴合当前帧
        auto t0 = std::chrono::steady_clock::now();
        std::vector<Detection> dets = run_sliced_infer(image);
        auto t1 = std::chrono::steady_clock::now();
        double infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // 绘制检测框（绿色高亮 + 置信度标签）
        for (const auto& d : dets) {
            char score_text[32];
            snprintf(score_text, sizeof(score_text), "%.2f", d.score);
            draw_modern_ui(image, d.box, cv::Scalar(0, 255, 0), score_text, true);
        }

        // 可选：把检测结果作为 ROS 消息发出去（后续如果接解码就方便了）
        if (!dets.empty()) {
            std_msgs::msg::String msg;
            char buf[128];
            snprintf(buf, sizeof(buf), "detected %zu barcodes", dets.size());
            msg.data = buf;
            publisher_->publish(msg);
        }

        char hud_text[120];
        snprintf(hud_text, sizeof(hud_text), "FPS: %.1f | Infer: %.1fms | Dets: %zu", 
            current_fps_, infer_ms, dets.size());
        cv::putText(image, hud_text, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 165, 255), 2, cv::LINE_AA);

        // 显示缩小一半，避免窗口太大
        cv::Mat display;
        cv::resize(image, display, cv::Size(image.cols / 2, image.rows / 2));
        cv::imshow("YOLOv8 Pure Detection (1280x720 + 1x2 Slicing)", display);
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