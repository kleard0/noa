from script.quantizer import Quantizer
from script.pruning import Pruning

from utils.utils import download_hf_model, check_model_folder, convert_hf_to_gguf, load_config



class Optimizer():
    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.hf_connector = self.config["hf_connector"]  
        self.hf_token = self.hf_connector["token"] 
        self.model_path = self.hf_connector["model_path"] 

        self.local_model = self.config["local_model"]  
        self.model_name = self.local_model["model_name"]
        self.model_dir = self.local_model["model_dir"]
        
        quantizer = Quantizer(config_path)

    def pre_run(self):
        download_hf_model(self)
        check_model_folder(self)
        convert_hf_to_gguf(self)

    def run_optimisation():
        """
        """      
            
            