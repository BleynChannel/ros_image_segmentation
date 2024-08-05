# Install & Setup

> **INSTALL.md WIP!**

## Local host

### Requirements
- [Ubuntu 20.04](https://ubuntu.com/)
- Nvidia graphics with CUDA >= 12.0, Cudnn >= 8.9 and TensorRT >= 8.6 (Check your GPU driver and CUDA version with `nvidia-smi`)
- [Python 3.10](https://www.python.org/downloads/)
- [ROS 1 Noetic](https://www.ros.org/)
- [PaddlePaddle >= 2.6](https://www.paddlepaddle.org.cn/install/)
- [Numpy < 2.0](https://numpy.org/)
- [ros_numpy](https://github.com/eric-wieser/ros_numpy)
- [OpenCV](https://opencv.org/)

```bash
git clone https://github.com/BleynChannel/ros_image_segmentation.git && cd ./ros_image_segmentation
python app/main.py --config=data/config.yaml
```

## Docker
```bash
git clone https://github.com/BleynChannel/ros_image_segmentation.git && cd ./ros_image_segmentation
docker-compose up --build
```

# For Developers
## Dev container
This repository contains the necessary files to build a Docker container for ROS Image Semantic Segmentation in development using Visual Studio Code.

### Prerequisites
Before you begin, ensure you have the following installed:

- Docker: [Install Docker](https://docs.docker.com/get-docker/)
- Visual Studio Code: [Download VS Code](https://code.visualstudio.com/)
- Visual Studio Code Remote - Containers extension: [Install the extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### Getting Started
To build and run this project in the development container, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/BleynChannel/ros_image_segmentation.git
```
2. Open the repository in Visual Studio Code:
```bash
code ros_image_segmentation
```
3. Press F1 to open the command palette and select `Remote-Containers: Open Folder in Container...`.
4. Select the root folder of the this repository.
5. To run the project, open the terminal in Visual Studio Code and run the following command:
```bash
cd ./app
python app/main.py --config=data/config.yaml
```

