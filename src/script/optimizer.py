from script.quantizer import Quantizer
from script.pruning import Pruning
from utils.utils import load_config, check_model_folder, download_hf_model


QUANTIZATION_OPTION [

    "F16",
    "Q8_0",
    "Q8_K",
    "Q6_K",
    "Q5_0",
    "Q5_K",
    "Q4_0",
    "Q4_1",      
    "Q4_K",
    "Q4_K_S",
    "Q4_K_M",
    "Q3_0",
    "Q3_K",
    "Q2_0",
    "Q2_K",
]
PRUNING_LEVELS = [
    0.0,   # 0%  
    0.1,   # 10%
    0.2,   # 20%
    0.3,   # 30%
    0.4,   # 40%
    0.5,   # 50%
    0.6,   # 60%
    0.7,   # 70%
    0.8,   # 80%
    0.85,  # 85%
    0.9,   # 90%  
]


class Optimizer():
    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.iteration = config["iteration"]
        
        self.quantizer = Quantizer(self.config)
        self.pruner = Pruning(self.config)
        self.model_name = self.config["model_name"]
        self.hf_token = config["HF_token"]
        
        self.model_file_name = str(self.model_name) + ".gguf"
		outfile = self.convert_hf
                     
        
    def run_optimisation(self, QUANTIZATION_OPTION, PRUNING_LEVELS):
        download_hf_model(self)
        check_model_folder(self)
        for quant, prune in zip(QUANTIZATION_OPTION, PRUNING_LEVELS):
            self.pruner.run(prune)
            self.quantizer.run()
            
            
            
            
            
            
            
            
            