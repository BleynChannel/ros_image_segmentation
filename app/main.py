import logging
import argparse
import signal

from src.config import AppConfig
from src.predictor import Predictor
from src.ros_pool import RosPool

is_stopped = False

def main(args):
    config = AppConfig(config_path=args.config_path)

    # Configure logging
    if config.log:
        logging.basicConfig(level=config.log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename=config.log_path)

    # Start ROS pool
    predictor = Predictor(config)
    ros_pool = RosPool(config, predictor)

    # Handle SIGINT and SIGTERM
    def stop_signal_handler(signal_number, stack_frame):
        global is_stopped

        if not is_stopped:
            is_stopped = True
            logging.info('Stopping...')
            ros_pool.stop()
        else:
            logging.info('Exiting...')
            exit(0)

    signal.signal(signal.SIGINT, stop_signal_handler)
    signal.signal(signal.SIGTERM, stop_signal_handler)

    ros_pool.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config.yaml', help='Path to the config file')
    args = parser.parse_args()

    main(args)
