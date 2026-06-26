#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>        // 仅用作 blobFromImage 预处理
#include <onnxruntime_cxx_api.h>  // 💡 微软官方推理引擎，彻底解决 OpenCV DNN 的崩溃问题
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

        // ================= 1. 初始化 ONNXRuntime 引擎 =================
        std::string model_path = "/home/zdh/ffs_ws/models/YOLOV8s_Barcode_Detection.onnx";
        
        try {
            // 💡 性能解锁：释放 i7 的多核算力！将单核改为多核并发推理
            session_options_.SetIntraOpNumThreads(6); 
            // 开启最高级别图优化，融合算子，极大降低延迟
            session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            
            // 加载模型
            session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
            RCLCPP_INFO(this->get_logger(), "✅ ONNXRuntime 引擎加载 YOLOv8 成功！(多核加速已开启)");
        } catch (const Ort::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "❌ ONNXRuntime 加载失败: %s", e.what());
            rclcpp::shutdown();
        }

        // ================= 2. 初始化 ZBar (格式白名单) =================
        scanner_.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_ENABLE, 0);
        scanner_.set_config(zbar::ZBAR_EAN13, zbar::ZBAR_CFG_ENABLE, 1);
        // 提高扫描密度：横向和纵向每隔 1 个像素就扫描一次，配合我们的多角度轮询，实现无死角
        scanner_.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_X_DENSITY, 1);
        scanner_.set_config(zbar::ZBAR_NONE, zbar::ZBAR_CFG_Y_DENSITY, 1);

        // ================= 3. 初始化 D415 硬件 =================
        rs2::config cfg;
        cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, 30);
        

        try {
            auto profile = pipeline_.start(cfg);
            auto sensor = profile.get_device().query_sensors()[1];

            // =================  硬件层抗反光 =================
            if (sensor.supports(RS2_OPTION_ENABLE_AUTO_EXPOSURE)) {
                sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 0);
            }
            //  曝光再压低 —— 宁可暗也不能过曝，过曝的条码信息是不可逆丢失
            // 反光场景的黄金原则：保留高光细节优先于保留暗部
            // sensor.set_option(RS2_OPTION_EXPOSURE, 60.0f);
            // sensor.set_option(RS2_OPTION_GAIN, 16.0f);

            // 开启背光补偿 (如果支持) —— 让相机自己处理高动态场景
            if (sensor.supports(RS2_OPTION_BACKLIGHT_COMPENSATION)) {
                sensor.set_option(RS2_OPTION_BACKLIGHT_COMPENSATION, 1);
            }
            // 关闭白平衡自动调整，防止反光区导致色调抽风
            if (sensor.supports(RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE)) {
                sensor.set_option(RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE, 0);
            }
            // ===================================================

            RCLCPP_INFO(this->get_logger(), "🚀 D415 YOLO+ORT+ZBar (终极满血版) 启动！");
        } catch (const rs2::error & e) {
            RCLCPP_ERROR(this->get_logger(), "相机启动失败: %s", e.what());
            rclcpp::shutdown();
        }

        // 💡 定时器：改为 10ms，让获取帧的频率高于相机 30fps 的物理极限，杜绝漏帧
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),
            std::bind(&BarcodeDetectorNode::process_frame, this)
        );
        last_time_ = std::chrono::steady_clock::now();
    }

private:
    struct BarcodeResult { cv::Rect rect; std::string data; };

    // ORT 变量
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<const char*> input_node_names_ = {"images"};
    std::vector<const char*> output_node_names_ = {"output0"};

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    rs2::pipeline pipeline_;
    zbar::ImageScanner scanner_;
    
    // 状态保持
    std::unordered_set<std::string> scanned_barcodes_;
    std::unordered_map<std::string, int> candidate_counts_;
    uint64_t total_frames_ = 0;
    std::vector<BarcodeResult> current_results_;
    
    // FPS 统计
    std::chrono::time_point<std::chrono::steady_clock> last_time_;
    double current_fps_ = 0.0;

    // ================= 视觉升级：绘制赛博朋克风 UI =================
    // 这个函数取代了枯燥的 cv::rectangle 实线框
    void draw_modern_ui(cv::Mat& img, const cv::Rect& r, const cv::Scalar& color, const std::string& label, bool is_locked) {
        int t = is_locked ? 3 : 2; // 如果已经成功解码(is_locked)，边框加粗
        int len = std::max(10, std::min(r.width, r.height) / 5); // 动态计算四角折线的长度

        // 绘制科技感四角折线 (雷达锁定风格)
        cv::line(img, cv::Point(r.x, r.y), cv::Point(r.x + len, r.y), color, t);
        cv::line(img, cv::Point(r.x, r.y), cv::Point(r.x, r.y + len), color, t);
        cv::line(img, cv::Point(r.x + r.width, r.y), cv::Point(r.x + r.width - len, r.y), color, t);
        cv::line(img, cv::Point(r.x + r.width, r.y), cv::Point(r.x + r.width, r.y + len), color, t);
        cv::line(img, cv::Point(r.x, r.y + r.height), cv::Point(r.x + len, r.y + r.height), color, t);
        cv::line(img, cv::Point(r.x, r.y + r.height), cv::Point(r.x, r.y + r.height - len), color, t);
        cv::line(img, cv::Point(r.x + r.width, r.y + r.height), cv::Point(r.x + r.width - len, r.y + r.height), color, t);
        cv::line(img, cv::Point(r.x + r.width, r.y + r.height), cv::Point(r.x + r.width, r.y + r.height - len), color, t);

        if (is_locked) {
            // 如果解码成功，在框内叠加一层半透明蒙版，增强视觉反馈
            cv::Mat overlay;
            img.copyTo(overlay);
            cv::rectangle(overlay, r, color, cv::FILLED);
            cv::addWeighted(overlay, 0.15, img, 0.85, 0, img);
            
            // 绘制带有底色的文字标签，防止背景杂乱导致看不清文字
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseLine);
            cv::Rect labelRect(r.x, r.y - labelSize.height - 10, labelSize.width + 10, labelSize.height + 10);
            cv::rectangle(img, labelRect, color, cv::FILLED);
            // 文字用黑色显示在彩色底色上，对比度最高
            cv::putText(img, label, cv::Point(r.x + 5, r.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
        }
    }

    // ============================================================
    //  严格校验 EAN-13，杜绝一切高强度滤波产生的杂纹误码
    // ============================================================
    bool validate_ean13(const std::string & code) {
        if (code.size() != 13) return false;
        for (char c : code) if (c < '0' || c > '9') return false;
        int sum = 0;
        for (int i = 0; i < 12; i++) {
            int d = code[i] - '0';
            sum += (i % 2 == 0) ? d : d * 3;
        }
        return ((10 - sum % 10) % 10) == (code[12] - '0');
    }

    // ============================================================
    //  无损图像旋转：旋转图像并自动扩大边界框
    // 防止条码的四个角被裁剪掉，边框用白色填充以给 ZBar 形成"静区"
    // ============================================================
    cv::Mat rotate_image_safely(const cv::Mat& src, double angle) {
        if (angle == 0.0) return src.clone();
        cv::Point2f center(src.cols / 2.0f, src.rows / 2.0f);
        cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
        cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), src.size(), angle).boundingRect2f();
        rot.at<double>(0, 2) += bbox.width / 2.0 - center.x;
        rot.at<double>(1, 2) += bbox.height / 2.0 - center.y;
        cv::Mat dst;
        // BORDER_CONSTANT 和 Scalar(255) 是灵魂，纯白填充给扫码算法提供了最完美的缓冲带
        cv::warpAffine(src, dst, rot, bbox.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255));
        return dst;
    }

    // ============================================================
    //  反光检测：找出图像中过曝的高光区域
    // 返回 true 表示存在超过 1% 的显著反光
    // ============================================================
    bool detect_glare(const cv::Mat & gray, cv::Mat & glare_mask) {
        cv::threshold(gray, glare_mask, 240, 255, cv::THRESH_BINARY);
        return ((double)cv::countNonZero(glare_mask) / (gray.rows * gray.cols)) > 0.01;
    }

    // ============================================================
    //  反光抑制方法 1：局部亮度归一化 (Retinex 思想简化版)
    // 用大核模糊估计光照，减去后得到"去光照"的反射图。对薄膜反光极度有效
    // ============================================================
    cv::Mat suppress_glare_retinex(const cv::Mat & gray) {
        cv::Mat gray_f, illum, reflect;
        gray.convertTo(gray_f, CV_32F, 1.0 / 255.0);
        cv::GaussianBlur(gray_f, illum, cv::Size(51, 51), 25); // 估计光照
        cv::log(gray_f + 0.01f, gray_f);
        cv::log(illum + 0.01f, illum);
        reflect = gray_f - illum; // 减去光照影响
        cv::Mat out;
        cv::normalize(reflect, reflect, 0, 255, cv::NORM_MINMAX);
        reflect.convertTo(out, CV_8U);
        return out;
    }

    // ============================================================
    // 💡 反光抑制方法 2：高光 inpaint 修复
    // 把过曝区域 mask 掉，用周边信息填补。代价高但对小面积强反光有效
    // ============================================================
    cv::Mat suppress_glare_inpaint(const cv::Mat & gray, const cv::Mat & glare_mask) {
        cv::Mat mask_dilated, out;
        cv::dilate(glare_mask, mask_dilated, cv::Mat(), cv::Point(-1, -1), 2);
        cv::inpaint(gray, mask_dilated, out, 3, cv::INPAINT_TELEA);
        return out;
    }

    // ============================================================
    // 💡 反光抑制方法 3：Top-hat 黑帽形态学 (提取暗细节)
    // 反光是亮区，条码的条纹是暗线 —— 黑帽能直接无视反光凸显暗条纹
    // ============================================================
    cv::Mat suppress_glare_tophat(const cv::Mat & gray) {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
        cv::Mat blackhat, out;
        cv::morphologyEx(gray, blackhat, cv::MORPH_BLACKHAT, kernel);
        cv::threshold(blackhat, out, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
        return out;
    }

    // ============================================================
    // 💡 策略 0：基础 CLAHE 增强锐化 (无反光/轻反光时默认启用)
    // ============================================================
    cv::Mat enhance_small_barcode(const cv::Mat & gray, double scale) {
        cv::Mat up, blur, sharp, out;
        cv::resize(gray, up, cv::Size(), scale, scale, cv::INTER_CUBIC);
        cv::GaussianBlur(up, blur, cv::Size(0, 0), 1.2);
        cv::addWeighted(up, 1.8, blur, -0.8, 0, sharp); // 图像锐化公式
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.5, cv::Size(16, 16));
        clahe->apply(sharp, out);
        return out;
    }

    // ============================================================
    // 💡 极速解码内核：只负责解码并校验，不涉及复杂的数学坐标计算
    // ============================================================
    std::string decode_roi(const cv::Mat & scan_gray) {
        if (scan_gray.empty() || !scan_gray.isContinuous()) return "";
        zbar::Image zbar_image(scan_gray.cols, scan_gray.rows, "Y800", (uchar *)scan_gray.data, scan_gray.cols * scan_gray.rows);
        if (scanner_.scan(zbar_image) <= 0) return "";
        for (auto symbol = zbar_image.symbol_begin(); symbol != zbar_image.symbol_end(); ++symbol) {
            std::string data = symbol->get_data();
            if (validate_ean13(data)) return data; // 命中且校验通过立即返回
        }
        return "";
    }

    // ============================================================
    // 💡 阶梯式抗反光流水线：一旦命中立即退出，保证系统极速
    // ============================================================
    std::string scan_with_antiglare(const cv::Mat & roi_gray, double scale) {
        cv::Mat glare_mask;
        bool has_glare = detect_glare(roi_gray, glare_mask);
        
        // 尝试 0：标准增强 (极速)
        std::string data = decode_roi(enhance_small_barcode(roi_gray, scale));
        if (!data.empty()) return data;
        
        // 如果没检测到强反光，就不浪费算力去跑后续的重度算法了
        if (!has_glare) return "";

        // 尝试 1：Retinex 去光照
        cv::Mat up1; cv::resize(suppress_glare_retinex(roi_gray), up1, cv::Size(), scale, scale, cv::INTER_CUBIC);
        if (!(data = decode_roi(up1)).empty()) return data;

        // 尝试 2：Top-hat 提暗线
        cv::Mat up2; cv::resize(suppress_glare_tophat(roi_gray), up2, cv::Size(), scale, scale, cv::INTER_CUBIC);
        if (!(data = decode_roi(up2)).empty()) return data;

        // 尝试 3：Inpaint 修复 (仅在反光面积<15%时使用，否则修复也是糊的)
        if (((double)cv::countNonZero(glare_mask) / (roi_gray.rows * roi_gray.cols)) < 0.15) {
            if (!(data = decode_roi(enhance_small_barcode(suppress_glare_inpaint(roi_gray, glare_mask), scale))).empty()) return data;
        }

        // 尝试 4：强对比度自适应二值化 (大力出奇迹)
        cv::Mat up3, bin; cv::resize(roi_gray, up3, cv::Size(), scale, scale, cv::INTER_CUBIC);
        cv::adaptiveThreshold(up3, bin, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 31, 10);
        return decode_roi(bin);
    }

    void process_frame() {
        rs2::frameset frames;

        // if (!pipeline_.poll_for_frames(&frames)) return;
        // rs2::video_frame color_frame = frames.get_color_frame();
        // if (!color_frame) return;

        // ================= 💡 修复：使用阻塞等待机制 =================
        try {
            // 最多等待 100ms。这能完美同步相机的 30fps，避免 CPU 空转
            frames = pipeline_.wait_for_frames(100); 
        } catch (const rs2::error& e) {
            // 如果预热阶段拿不到画面超时了，必须调用 waitKey 给窗口渲染的机会！
            cv::waitKey(1);
            return;
        }

        rs2::video_frame color_frame = frames.get_color_frame();
        if (!color_frame) {
            cv::waitKey(1); // 同样，拿不到彩色帧也要防死锁
            return;
        }
        // ==============================================================

        // ================= 💡 FPS 平滑计算 =================
        auto now = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(now - last_time_).count();
        last_time_ = now;
        if (dt > 0.0 && dt < 1.0) {
            if (current_fps_ == 0.0) current_fps_ = 1.0 / dt;
            else current_fps_ = (current_fps_ * 0.9) + ((1.0 / dt) * 0.1); // 一阶低通滤波
        }

        cv::Mat image(cv::Size(1280, 720), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        total_frames_++;

        // 每两帧做一次深度学习推理，这是机器人视觉降低延迟的黄金法则
        if (total_frames_ % 2 == 0) {
            current_results_.clear();

            // ================= 新增：全局轻量级抗反光 (给 YOLO 戴上墨镜) =================
            // // 转换到 HSV 色彩空间，只处理亮度通道，防止色彩失真导致 YOLO 不认识
            // cv::Mat hsv;
            // cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
            // std::vector<cv::Mat> channels;
            // cv::split(hsv, channels);

            // // 应用 CLAHE (限制对比度自适应直方图均衡化)
            // // 参数 2.0 和 8x8 是黄金比例：能压制塑料膜高光，同时凸显暗部条码
            // cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
            // clahe->apply(channels[2], channels[2]); // 只处理 V(亮度) 通道

            // // 合并并转回 BGR 彩色图，此时图像的高光会被压制，条码纹理会变清晰
            // cv::merge(channels, hsv);
            // cv::cvtColor(hsv, image, cv::COLOR_HSV2BGR);
            // // ==============================================================================

            // ================= YOLO 检测准确率的核心：Letterbox 等比例缩放 =================
            // 之前的粗暴 resize 会把长条形的条码压成正方形，破坏特征
            // Letterbox 会将图像等比例缩小，并在不足的地方填充黑色边框，保持真实物理比例
            float scale_ratio = std::min(640.0f / image.cols, 640.0f / image.rows);
            int new_w = std::round(image.cols * scale_ratio);
            int new_h = std::round(image.rows * scale_ratio);
            
            cv::Mat resized;
            cv::resize(image, resized, cv::Size(new_w, new_h));
            cv::Mat letterbox = cv::Mat::zeros(640, 640, CV_8UC3); // 黑底画布
            int pad_w = (640 - new_w) / 2;
            int pad_h = (640 - new_h) / 2;
            // 将缩小后的原图贴到黑底画布的正中间
            resized.copyTo(letterbox(cv::Rect(pad_w, pad_h, new_w, new_h)));

            // 预处理与推理
            cv::Mat blob;
            cv::dnn::blobFromImage(letterbox, blob, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);

            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            std::vector<int64_t> input_dims = {1, 3, 640, 640};
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)blob.data, blob.total(), input_dims.data(), input_dims.size());

            auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_node_names_.data(), &input_tensor, 1, output_node_names_.data(), 1);
            float* raw_output = output_tensors[0].GetTensorMutableData<float>();
            
            std::vector<cv::Rect> boxes;
            std::vector<float> scores;

            // YOLOv8 锚框解析 (8400个网格点)
            for (int i = 0; i < 8400; ++i) {
                // 第4、5通道分别是 barcode 和 qrcode 的置信度
                float max_score = std::max(raw_output[4 * 8400 + i], raw_output[5 * 8400 + i]);

                if (max_score > 0.45) { 
                    float xc = raw_output[0 * 8400 + i];
                    float yc = raw_output[1 * 8400 + i];
                    float w  = raw_output[2 * 8400 + i];
                    float h  = raw_output[3 * 8400 + i];

                    // 💡 坐标逆映射：扣除 Letterbox 的黑边，并按比例还原回 1280x720 的真实坐标
                    int left = int((xc - 0.5 * w - pad_w) / scale_ratio);
                    int top = int((yc - 0.5 * h - pad_h) / scale_ratio);
                    int width = int(w / scale_ratio);
                    int height = int(h / scale_ratio);

                    boxes.push_back(cv::Rect(left, top, width, height));
                    scores.push_back(max_score);
                }
            }

            // 非极大值抑制 (消除重叠的多余检测框)
            std::vector<int> indices;
            cv::dnn::NMSBoxes(boxes, scores, 0.45, 0.4, indices);

            for (int idx : indices) {
                cv::Rect r = boxes[idx];

                // 💡 固定像素的精准外扩 (Padding)：确保留出 ZBar 需要的安静白区，且不包含多余环境杂色
                int pad = 20;
                r.x = std::max(0, r.x - pad);
                r.y = std::max(0, r.y - pad);
                r.width = std::min(image.cols - r.x, r.width + 2 * pad);
                r.height = std::min(image.rows - r.y, r.height + 2 * pad);
                
                // 画未确认的浅蓝色赛博锁定瞄准框 (只有框，没有文字)
                draw_modern_ui(image, r, cv::Scalar(255, 150, 0), "", false);

                cv::Mat roi_gray;
                cv::cvtColor(image(r), roi_gray, cv::COLOR_BGR2GRAY);

                // ================= 💡 核心升级：多角度轮询攻击 =================
                // 每 30 度转一次，加上 ZBar 自带的容错，构成 360 度绝对无死角天罗地网！
                std::vector<double> angles = {0, 30, -30, 60, -60, 90};
                for (double angle : angles) {
                    cv::Mat rotated_roi = rotate_image_safely(roi_gray, angle);
                    
                    // 动态自适应放大倍数 (将目标强行拉伸到 ZBar 的最佳工作区 600px 左右)
                    double scale = std::max(2.0, std::min(8.0, 600.0 / rotated_roi.cols));
                    
                    // 送入抗反光管线解码
                    std::string data = scan_with_antiglare(rotated_roi, scale);

                    if (!data.empty()) {
                        candidate_counts_[data]++;
                        // ===== 解码成功，进入多帧投票机制 =====
                        if (candidate_counts_[data] >= 2) { 
                            current_results_.push_back({r, data});
                            // 确保不重复发布同一个条码
                            if (scanned_barcodes_.find(data) == scanned_barcodes_.end()) {
                                scanned_barcodes_.insert(data);
                                std_msgs::msg::String msg; msg.data = data; publisher_->publish(msg);
                                RCLCPP_INFO(this->get_logger(), "🟢 完美捕获: %s (ORT推理, %.0f° 纠偏成功)", data.c_str(), angle);
                            }
                        }
                        // 只要当前检测框内成功解出了条码，直接跳出角度循环，不再浪费算力
                        break; 
                    }
                }
            }
        }

        // 定期清理计票器，防止内存泄漏和幽灵条码
        if (total_frames_ % 100 == 0) candidate_counts_.clear();
        
        // ================= 💡 绘制最终结果 =================
        for (const auto & res : current_results_) {
            // 绘制已确认的绿色赛博锁定框，附带半透明背景和文字标签
            draw_modern_ui(image, res.rect, cv::Scalar(0, 255, 0), res.data, true);
        }
        
        // 渲染 FPS
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