ARG ROS_DISTRO=noetic

FROM ros:$ROS_DISTRO

RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> ~/.bashrc

RUN apt-get update && apt-get install -y python3-pip && \
	apt-get install ffmpeg libsm6 libxext6 -y && \
	apt-get clean && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install --upgrade pip

RUN pip install opencv-python
# RUN pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
RUN pip install onnxruntime
RUN pip install numpy rosnumpy pyyaml

WORKDIR /app
COPY /app /app