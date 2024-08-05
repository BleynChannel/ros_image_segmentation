FROM bleyn/paddle-ros:2.6.1-gpu-cuda12.0-cudnn8.9-trt8.6-noetic

RUN pip install 'numpy<2' rosnumpy opencv-python pyyaml

WORKDIR /root

# DEVELOPER

# # clone paddleseg source
# ARG PADDLESEG_BRANCH="release/v2.0.0-rc"
# RUN git clone -b develop https://github.com/PaddlePaddle/PaddleSeg /paddleseg
# WORKDIR /paddleseg
# RUN pip install -r requirements.txt

# RUN pip install 'numpy<2'
# RUN python setup.py install --install-lib /usr/local/lib/python3.10/dist-packages/

# clone paddleslim source
# ARG PADDLESLIM_BRANCH="release/v2.0.0-rc"
# RUN git clone https://github.com/PaddlePaddle/PaddleSlim.git /paddleslim
# WORKDIR /paddleslim
# RUN pip install -r requirements.txt
# RUN python setup.py install --install-lib /usr/local/lib/python3.10/dist-packages/

# WORKDIR /paddleseg/deploy/slim/act
# RUN wget https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.9.1/deploy/slim/act/configs/ppliteseg/ppliteseg_qat.yaml
# RUN wget https://paddleseg.bj.bcebos.com/deploy/slim_act/ppliteseg/save_quant_model_qat.zip
# RUN unzip ./save_quant_model_qat.zip
# RUN wget https://bj.bcebos.com/v1/paddle-slim-models/data/mini_cityscapes/mini_cityscapes.tar
# RUN mkdir data && tar -xvf mini_cityscapes.tar -C data/
# RUN mv data/mini_cityscapes data/cityscapes