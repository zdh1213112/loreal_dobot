# D435 画质与转盘条形码对比测试

独立程序：[scripts/d435_barcode_quality_lab.py](scripts/d435_barcode_quality_lab.py)。它只打开 D435 彩色相机并运行与 V3 相同的条形码候选检测器；**不控制转盘、机械臂或夹爪，也不发布 ROS 消息**。RealSense 通常不能让两个程序同时占用同一相机，因此先正常停止 V3，再运行此工具。不要为了试相机在机械臂运动中强制关闭 V3。要测旋转场景，转盘须由现有的独立安全方式操作。

先查询这台相机实际支持的 BGR8 模式和选项范围：

```bash
/home/zdh/miniconda3/envs/ffs_ros/bin/python \
  /home/zdh/ffs_ws/src/dobot_nova5_driver/scripts/d435_barcode_quality_lab.py \
  --list-profiles
```

每次换一个**新的**输出目录，分别采集同一个物料、相同光照和转速下的结果：

```bash
LAB=/home/zdh/ffs_ws/src/dobot_nova5_driver/scripts/d435_barcode_quality_lab.py
PY=/home/zdh/miniconda3/envs/ffs_ros/bin/python

$PY "$LAB" --mode native --duration 20 --output /home/zdh/ffs_ws/d435_lab_native_01
$PY "$LAB" --mode auto --profile 1280x720@30 --duration 20 \
  --output /home/zdh/ffs_ws/d435_lab_auto_01
$PY "$LAB" --mode manual --profile 1280x720@30 --exposure 70 --gain 128 \
  --duration 20 --output /home/zdh/ffs_ws/d435_lab_manual70_01
$PY "$LAB" --mode manual --profile 1920x1080@30 --exposure 50 --gain 160 \
  --duration 20 --output /home/zdh/ffs_ws/d435_lab_1080p_01
```

最后一组仅在 `--list-profiles` 确认本机支持 `1920x1080@30` BGR8 时运行。分辨率切换后，工具将生产程序的 760×440 检测细节窗口移动到画面相同的相对中心，并保留每 6 帧一次的整幅检测；这样高分辨率试验不会继续看错 1280×720 的旧像素坐标。

`native` 只请求相机默认彩色流，不设置分辨率、帧率、自动曝光、曝光、增益或锐度；它表示“本程序不改相机选项”，**不保证恢复出厂值**，因为设备可能保留上一个程序的设置。运行时会显示实际选中的模式和读回的相机选项。`auto` 固定所选分辨率/FPS，开启 AE 并关闭“允许 AE 降低帧率”；`manual` 固定所选模式、曝光和增益。可选 `--sharpness 数值`，先用 `--list-profiles` 查看范围。手动/自动测试结束时程序尽力恢复启动前的相机选项；仍应在重新启动 V3 后检查其启动日志。

画面按键：`q`/Esc 退出，`r` 换分辨率，`f` 换同分辨率下的 FPS，`a` 切换 AE/手动，`[`/`]` 调低/调高曝光，`,`/`.` 调低/调高增益，`s` 立即保存一张原始 PNG。鼠标左键拖选条形码接受 ROI，右键清除；ROI 只限制候选**中心**是否被接受，不裁切模型输入。原生模式不能在本进程内切换分辨率或 AE；换模式应重新运行。无 GUI 环境可加 `--headless`。

输出中的 `raw_*.png` 是未画框的彩色帧，以 PNG 无损保存，**不是 V3 的 JPEG 预览截图**；若相机原生流本身选中了 MJPEG，先前的相机压缩当然无法由 PNG 逆转。`frames.csv` 记录每帧曝光、增益、候选来源/分数、连续命中数、检测耗时和候选区清晰度；`summary.json` 汇总实际交付帧率、候选帧数和确认次数。黄色框=单帧候选，绿色框/`CONFIRMED`=达到与 V3 相同的稳定命中门槛。`delivered_fps` 包含检测耗时，是此程序实际处理速度，不应当误称传感器物理帧率；可以同时看帧号、相机时间戳和 `detect_ms`。不同画面内容的 Laplacian 清晰度数值不能单独当作光学质量结论，优先比较同一标签位置的 `candidate_laplacian_var` 和原始 PNG 中条纹的可辨度。

塑封膜的镜面反射无法只靠提高分辨率消除。尽量使用均匀的漫射补光、让相机或灯与膜的镜面反射角错开；在允许的安装条件下，可尝试偏振光源/镜头偏振片。旋转时优先试较短曝光并用补光或适度增益补亮；曝光过长会拖成条纹，增益太高又会增加噪声。任何安装角度或照明改变后都要重新确认识别效果，不要把画面变锐视为“已无误检”。
