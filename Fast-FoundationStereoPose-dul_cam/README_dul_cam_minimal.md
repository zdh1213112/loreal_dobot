# Fast-FoundationStereoPose dul_cam 精简版

这个目录只保留 `dul_cam` 运行相关代码，不包含大模型权重、数据集、demo 图片视频和 Git 历史。

## 已保留

- `dul_cam/`：D405/D435/Nova5 相关脚本与配置
- `core/`、`Utils.py`：FoundationStereo 相关代码依赖
- `SAM2_streaming/sam2/`、`SAM2_streaming/configs/`：SAM2 代码与配置
- `weights/23-36-37/cfg.yaml`：FFS 模型配置
- `requirements.txt`、`.gitignore`、`README.md`、`LICENSE.txt`

## 需要本地补齐的大文件/本机路径

- `weights/23-36-37/model_best_bp2_serialize.pth`
- `SAM2_streaming/checkpoints/sam2.1/sam2.1_hiera_small.pt`
- YOLO 权重路径，例如代码或 `dul_cam/d435_nova5_rrt_config.yaml` 中的 `/home/zdh/.../weights/best.pt`
- RealSense 设备序列号、Nova5 驱动路径、机器人 IP 等本机配置

当前这份本地目录已从原项目复制了 FFS 权重和 SAM2 checkpoint，能用于本机运行；它们仍会被 `.gitignore` 忽略，不会上传到 Git。

## 建议上传 Git

```bash
git init
git add .
git status --short
git commit -m "Add dul_cam minimal project"
```
