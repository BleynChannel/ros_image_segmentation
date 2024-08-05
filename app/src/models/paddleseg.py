from .model import Model
import yaml
import os
import logging
import numpy as np

from paddle.inference import create_predictor, PrecisionType
from paddle.inference import Config as PredictConfig

from src.color_scheme import cityscapes_color_scheme

class PaddleSegConfig:
    def __init__(self, path: str):
        with open(path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
            
            self.model_filename = self.config['model_filename']
            self.params_filename = self.config['params_filename']
            self.color_scheme = self.config['color_scheme']
            self.precision = self.config['precision']
            self.cpu_threads = self.config['cpu']['threads']
            self.use_mkldnn = self.config['cpu']['use_mkldnn']
            self.use_trt = self.config['gpu']['use_trt']


def load_predictor(config: PaddleSegConfig, model_path: str, device: str = 'gpu'):
    """
    load predictor func
    """
    rerun_flag = False
    model_file = os.path.join(model_path, config.model_filename)
    params_file = os.path.join(model_path, config.params_filename)
    pred_cfg = PredictConfig(model_file, params_file)
    pred_cfg.enable_memory_optim()
    pred_cfg.switch_ir_optim(True)
    if device == "gpu":
        pred_cfg.enable_use_gpu(100, 0)
    else:
        pred_cfg.disable_gpu()
        pred_cfg.set_cpu_math_library_num_threads(config.cpu_threads)
        if config.use_mkldnn:
            pred_cfg.enable_mkldnn()
            if config.precision == "int8":
                # Please ensure that the quantized ops during inference are the same as
                # the ops set in the qat training configuration file
                pred_cfg.enable_mkldnn_int8({"conv2d", "depthwise_conv2d"})

    if config.use_trt:
        # To collect the dynamic shapes of inputs for TensorRT engine
        dynamic_shape_file = os.path.join(model_path, "dynamic_shape.txt")
        if os.path.exists(dynamic_shape_file):
            pred_cfg.enable_tuned_tensorrt_dynamic_shape(dynamic_shape_file,
                                                        True)
            # pred_cfg.exp_disable_tensorrt_ops(["reshape2"])
            print("trt set dynamic shape done!")
            precision_map = {
                "fp16": PrecisionType.Half,
                "fp32": PrecisionType.Float32,
                "int8": PrecisionType.Int8
            }
            pred_cfg.enable_tensorrt_engine(
                workspace_size=1 << 30,
                max_batch_size=1,
                min_subgraph_size=4,
                precision_mode=precision_map[config.precision],
                use_static=True,
                use_calib_mode=False, )
        else:
            pred_cfg.disable_gpu()
            pred_cfg.set_cpu_math_library_num_threads(10)
            pred_cfg.collect_shape_range_info(dynamic_shape_file)
            print("Start collect dynamic shape...")
            rerun_flag = True

    pred_cfg.exp_disable_tensorrt_ops(["reshape2"])
    # pred_cfg.delete_pass("gpu_cpu_map_matmul_v2_to_mul_pass")
    # pred_cfg.delete_pass("delete_quant_dequant_linear_op_pass")
    # pred_cfg.delete_pass("delete_weight_dequant_linear_op_pass")
    predictor = create_predictor(pred_cfg)
    return predictor, rerun_flag

class PaddleSeg(Model):
    def __init__(self, model_path: str, device = 'gpu'):
        super().__init__(model_path, device)
        self.config = PaddleSegConfig(model_path + '/config.yaml')

        if self.config.color_scheme == 'cityscapes':
            self.color_scheme = cityscapes_color_scheme
        else:
            raise ValueError(f'Color scheme {self.config.color_scheme} not supported')
        
        self.predictor, self.rerun_flag = load_predictor(self.config, model_path, device)

        self.input_names = self.predictor.get_input_names()
        self.input_handle = self.predictor.get_input_handle(self.input_names[0])
        self.output_names = self.predictor.get_output_names()
        self.output_handle = self.predictor.get_output_handle(self.output_names[0])

    def predict(self, inputs: np.ndarray[np.float32]) -> np.ndarray[np.uint8]:
        self.input_handle.reshape(inputs.shape)
        self.input_handle.copy_from_cpu(inputs)

        self.predictor.run()
        results = self.output_handle.copy_to_cpu()
        if self.rerun_flag:
            logging.warning(
                "***** Collect dynamic shape done, Please rerun the program to get correct results. *****"
            )
            exit(0)
        
        return np.asarray(results, dtype=np.uint8)