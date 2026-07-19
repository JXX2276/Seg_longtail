# Seg_longtail

## 1. 实验内容与模型流程

本项目在 Windows 单卡 16 GB 显存环境中，从公开预训练权重开始，完成 nuImages 四类的自动 Mask 生成与实例分割训练。

目标类别：

```text
0  movable_object.barrier
1  movable_object.debris
2  movable_object.pushable_pullable
3  movable_object.trafficcone
```

其中 `pushable_pullable` 作为数据量较多的主类，`barrier`、`debris` 和 `trafficcone` 作为重点增强的长尾类。

```text
nuImages 原始图片
  → Grounding DINO Tiny：低阈值开放词汇检测，生成高召回候选框
  → Qwen2.5-VL-7B-Instruct（BF16）：判断候选区域的语义类别，过滤明显误检
  → SAM2.1 Hiera Base Plus：以候选框为提示，生成实例多边形 Mask
  → 数据整理：模型生成的训练 Mask + nuImages 官方真值验证集
  → YOLO11s-seg：作为学生模型训练
  → 最终模型：对新图片输出类别、检测框和实例 Mask
```

| 阶段 | 主要输出 | 说明 |
|---|---|---|
| Grounding DINO | `workspace/teacher/candidates.jsonl` | 候选框、类别文本和检测分数 |
| Qwen | `workspace/teacher/verified.jsonl` | 每个候选区域的语义判断 |
| SAM2.1 | `workspace/teacher/segmented.jsonl` | 原始 Mask 多边形和 Mask 分数 |
| 数据整理 | `workspace/dataset/labels/train/*.txt` | 实际用于训练的 YOLO Segmentation Mask |
| YOLO11s-seg | `workspace/pipeline/final_selected.pt` | 最终四类实例分割模型 |

## 2. 环境配置

```powershell
uv sync --cache-dir .cache\uv --python 3.11 --no-python-downloads --no-dev
```

## 3. 数据集与模型预训练权重下载及目录

从 [nuImages 官方下载页面](https://www.nuscenes.org/nuimages) 登录并下载：

```text
nuImages v1.0 All Samples
nuImages v1.0 All Metadata
```

模型预训练权重官方链接：

- [Grounding DINO Tiny（IDEA-Research/grounding-dino-tiny）](https://huggingface.co/IDEA-Research/grounding-dino-tiny)
- [Qwen2.5-VL-7B-Instruct（Qwen/Qwen2.5-VL-7B-Instruct）](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [SAM2.1 Hiera Base Plus（facebook/sam2.1-hiera-base-plus）](https://huggingface.co/facebook/sam2.1-hiera-base-plus)
- [YOLO11s-seg（yolo11s-seg.pt）](https://huggingface.co/Ultralytics/YOLO11/blob/main/yolo11s-seg.pt)

模型无需手动逐个下载，第 4 节的下载脚本会将它们放入 `models`。所有下载、缓存和实验输出均位于当前项目内：

```text
Seg_longtail\
├─ .cache\
├─ .venv\
├─ configs\
├─ datasets\
│  ├─ nuimages-v1.0-all-samples\
│  └─ nuimages-v1.0-all-metadata\
│     └─ v1.0-train\
│        ├─ sample_data.json
│        ├─ object_ann.json
│        └─ category.json
├─ models\
│  ├─ grounding-dino-tiny\
│  ├─ Qwen2.5-VL-7B-Instruct\
│  ├─ sam2.1-hiera-base-plus\
│  └─ yolo11s-seg.pt
├─ scripts\
├─ src\
└─ workspace\
```

## 4. 运行与训练指令

下载四个预训练模型：

```powershell
.\.venv\Scripts\python.exe scripts\download_models.py
```

从原始图片开始处理 `configs/teacher_500.yaml` 指定的 500 个 sample，依次运行 Grounding DINO、Qwen 和 SAM2.1，并建立训练集与验证集：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_teacher_500.ps1
```

启动完整三阶段训练：

```powershell
.\.venv\Scripts\python.exe scripts\train_pipeline.py
```

| 阶段 | 初始化权重 | 训练设置 | 选模规则 |
|---|---|---|---|
| Stage 1 | `models/yolo11s-seg.pt` | 1024，batch 16，10 epoch，冻结前 10 层 | Ultralytics best |
| Stage 2 | Stage 1 最优权重 | 类别均衡，1024，batch 16，8 epoch | 长尾宏平均 Recall 最多下降 2 个百分点，再选择 Precision 较高者 |
| Stage 3 | Stage 2 选中权重 | 长尾强采样，1280，batch 8，12 epoch | 最大化长尾 Box/Mask 宏平均 Recall |

训练过程记录到 W&B group `stage1-stage2-stage3`，同时保存在 `workspace/pipeline`。最终权重为：

```text
workspace/pipeline/final_selected.pt
```

使用最终模型推理图片并保存可视化结果及多边形 Mask：

```powershell
.\.venv\Scripts\yolo.exe segment predict model=workspace\pipeline\final_selected.pt source=path\to\images imgsz=1280 conf=0.05 save=True save_txt=True save_conf=True project=workspace\inference name=predict
```

推理结果位于：

```text
workspace/inference/predict/          带 Mask 的可视化图片
workspace/inference/predict/labels/   YOLO Segmentation 多边形 Mask
```

