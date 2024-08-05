# ROS Image Semantic Segmentation

<div align="center">
	<a href="">📘Documentation</a> | <a href="">💻Install</a> | <a href="">🐼Model Zoo</a>
</div>

> **README.md WIP**

# Introduction
**ROS Image Semantic Segmentation** - algorithm for semantic segmentation on images from ROS camera stream. This algorithm based on PP-LiteSeg and connected with [ROS 1 Noetic](https://www.ros.org).

# Getting Started

For a step-by-step walkthrough of the installation process, please refer to our [Installation Guide](https://github.com/BleynChannel/2d_segmentation/blob/main/INSTALL.md). This guide provides detailed instructions to help you get up and running quickly.

# Tools

For build TensorRT cache for PP-LiteSeg model you can use command `python tools/paddleseg_trt_generate.py`. For example:
```bash
python tools/paddleseg_trt_generate.py --model_path=data/models/paddleseg --batch_size=1 --width=2048 --height=1024
```

**Args for `tools/paddleseg_trt_generate.py`:**
|Argument      |Description                  |Default|Type|
|--------------|-----------------------------|-------|----|
|`--model_path`|Path to model directory      |       |str |
|`--batch_size`|Batch size use for build     |1      |int |
|`--width`     |Width of model use for build |2048   |int |
|`--height`    |Height of model use for build|1024   |int |

# Model Zoo and Benchmark

|Model        |mAP    |Mem (GB)|Inf time (fps)|Download     |
|-------------|-------|--------|--------------|-------------|
|PP-LiteSeg   |       |        |              |[model](), [config]()|

------------------------

# Credit
This project based on [PaddleSeg API](https://github.com/PaddlePaddle/PaddleSeg).

# Licence
MIT