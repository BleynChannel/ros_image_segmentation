import argparse
import yaml
import os
import numpy as np

from paddle.inference import create_predictor, PrecisionType
from paddle.inference import Config as PredictConfig

class PaddleSegConfig:
    def __init__(self, path: str):
        with open(path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
            
            self.model_filename = self.config['model_filename']
            self.params_filename = self.config['params_filename']
            self.precision = self.config['precision']

def main(args):
    config = PaddleSegConfig(os.path.join(args.model_path, 'config.yaml'))
    inputs = np.zeros((args.batch_size, 3, args.height, args.width), dtype=np.float32)

    # load predictor
    trt_cache = False
    model_file = os.path.join(args.model_path, config.model_filename)
    params_file = os.path.join(args.model_path, config.params_filename)
    pred_cfg = PredictConfig(model_file, params_file)
    pred_cfg.enable_memory_optim()
    pred_cfg.switch_ir_optim(True)
    pred_cfg.enable_use_gpu(100, 0)

    # To collect the dynamic shapes of inputs for TensorRT engine
    dynamic_shape_file = os.path.join(args.model_path, "dynamic_shape.txt")
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
        
        trt_cache = True
    else:
        pred_cfg.disable_gpu()
        pred_cfg.set_cpu_math_library_num_threads(10)
        pred_cfg.collect_shape_range_info(dynamic_shape_file)
        print("Start collect dynamic shape...")

    pred_cfg.exp_disable_tensorrt_ops(["reshape2"])
    # pred_cfg.delete_pass("gpu_cpu_map_matmul_v2_to_mul_pass")
    # pred_cfg.delete_pass("delete_quant_dequant_linear_op_pass")
    # pred_cfg.delete_pass("delete_weight_dequant_linear_op_pass")
    predictor = create_predictor(pred_cfg)

    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])
    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])

    input_handle.reshape(inputs.shape)
    input_handle.copy_from_cpu(inputs)

    predictor.run()

    if trt_cache:
        print("***** Collect TensorRT cache done! *****")
    else:
        print("***** Collect dynamic shape done, Please rerun the program to get correct results. *****")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Path to the model directory')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size of the input image')
    parser.add_argument('--width', type=int, default=2048, help='Width of the input image')
    parser.add_argument('--height', type=int, default=1024, help='Height of the input image')
    args = parser.parse_args()

    main(args)