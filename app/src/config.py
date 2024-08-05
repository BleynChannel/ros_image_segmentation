import yaml

class CameraConfig:
    def __init__(self, camera_element):
        self.name: str = camera_element['name']
        self.raw_topic: str = camera_element['raw_topic']
        self.segmented_image_topic: str = camera_element['segmented_image_topic']
        self.overlayed_image_topic: str = camera_element['overlayed_image_topic']

class AppConfig:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
            
            self.log: bool = self.config['log']['active'] # True or False
            self.log_level: str = self.config['log']['level'] # 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
            self.log_path: str = self.config['log']['path']

            # Model
            self.model_type: str = self.config['model']['type']
            self.model_path: str = self.config['model']['path']
            self.device: str = self.config['model']['device'] # 'cpu' or 'gpu'

            self.predict_mode: str = self.config['predict_mode'] # 'single', 'batch'
            self.benchmark: bool = self.config['benchmark'] # True or False

            # ROS
            self.node_name: str = self.config['ros']['node_name']
            self.cameras: list[CameraConfig] = [CameraConfig(camera_element) for camera_element in self.config['ros']['cameras']]
