from huggingface_hub import snapshot_download, login
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
import yaml
from utils.utils import load_config
from pathlib import Path
import llama_cpp # C'est la librairie installée

class Quantizer():
    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.model_path = self.config["model_path"]
        self.model_name = self.config["model_name"]
        self.hf_token = self.config["HF_token"]
        
        self.quant_option = self.config["quantization_level"] # ex: "Q4_K_M"
        os.environ['HF_TOKEN'] = self.hf_token
        
        self.quant_map = {
            "Q4_0": llama_cpp.LLAMA_FTYPE_MOSTLY_Q4_0,
            "Q4_1": llama_cpp.LLAMA_FTYPE_MOSTLY_Q4_1,
            "Q5_0": llama_cpp.LLAMA_FTYPE_MOSTLY_Q5_0,
            "Q5_1": llama_cpp.LLAMA_FTYPE_MOSTLY_Q5_1,
            "Q8_0": llama_cpp.LLAMA_FTYPE_MOSTLY_Q8_0,
            "Q2_K": llama_cpp.LLAMA_FTYPE_MOSTLY_Q2_K,
            "Q3_K_S": llama_cpp.LLAMA_FTYPE_MOSTLY_Q3_K_S,
            "Q3_K_M": llama_cpp.LLAMA_FTYPE_MOSTLY_Q3_K_M,
            "Q3_K_L": llama_cpp.LLAMA_FTYPE_MOSTLY_Q3_K_L,
            "Q4_K_S": llama_cpp.LLAMA_FTYPE_MOSTLY_Q4_K_S,
            "Q4_K_M": llama_cpp.LLAMA_FTYPE_MOSTLY_Q4_K_M,
            "Q5_K_S": llama_cpp.LLAMA_FTYPE_MOSTLY_Q5_K_S,
            "Q5_K_M": llama_cpp.LLAMA_FTYPE_MOSTLY_Q5_K_M,
            "Q6_K": llama_cpp.LLAMA_FTYPE_MOSTLY_Q6_K,
        }

    def quantize_gguf(self, input_gguf):
        input_gguf = str(input_gguf)
        # On construit le nom de sortie
        output_gguf = f"{self.model_name}_{self.quant_option}.gguf"
        
        print("⚙️  Quantization GGUF via llama-cpp-python")
        print(f">>> Input: {input_gguf}")
        print(f">>> Output: {output_gguf}")
        print(f">>> Type: {self.quant_option}")

        # Vérification que le type de quantification existe
        if self.quant_option not in self.quant_map:
            raise ValueError(f"❌ Type de quantification inconnu : {self.quant_option}")

        ftype = self.quant_map[self.quant_option]

        try:
            # Configuration des paramètres de quantification
            # allow_requantize=True permet de re-quantifier un modèle déjà quantifié
            params = llama_cpp.llama_model_quantize_params(
                nthread=os.cpu_count(),
                ftype=ftype,
                allow_requantize=True,
                quantize_output_tensor=True,
                only_copy=False,
                pure=False,
                keep_split=False 
            )

            # Appel direct à la fonction C
            # Note: il faut encoder les chemins en bytes (utf-8) pour le C
            ret = llama_cpp.llama_model_quantize(
                input_gguf.encode("utf-8"),
                output_gguf.encode("utf-8"),
                params
            )

            if ret != 0:
                raise Exception(f"Code erreur retourné par llama.cpp: {ret}")

            print(f"✔️ Modèle quantifié généré : {output_gguf}")
            return output_gguf

        except Exception as e:
            print("❌ Erreur lors de la quantization :", e)
            raise

    def run(self, input_file):
        # Plus besoin d'ensure_llama_quantize
        return self.quantize_gguf(input_file)