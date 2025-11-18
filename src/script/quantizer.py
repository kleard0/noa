from huggingface_hub import snapshot_download, login, notebook_login
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
import subprocess
import yaml
from utils.utils import load_config
from pathlib import Path


class Quantizer():
    def __init__(self, config_path):

        self.config = load_config(config_path)
        self.model_path = self.config["model_path"]
        self.model_name = self.config["model_name"]
        os.environ['HF_TOKEN'] = self.config["HF_token"]
        
        
    def quantize_gguf(self, input_gguf, quant_option):
        input_gguf = str(input_gguf)
        output_gguf = str(input_gguf) + str(quant_option) + ".gguf"

        llama_quantize_bin = "llama.cpp/build/bin/llama-quantize"

        cmd = [
            llama_quantize_bin,
            input_gguf,
            self.model_name ,
            qtype
        ]

        print("⚙️  Quantization GGUF")
        print(">>>", " ".join(cmd))

        try:
            subprocess.check_call(cmd)
            print(f"✔️ Modèle quantifié généré : {output_gguf}")
        except subprocess.CalledProcessError as e:
            print("❌ Erreur lors de la quantization :", e)
            raise
        
    def ensure_llama_quantize(self, llama_cpp_dir="llama.cpp"):

        llama_cpp_dir = Path(llama_cpp_dir).resolve()
        build_dir = llama_cpp_dir / "build"
        bin_path = build_dir / "bin" / "llama-quantize"

        # 1️⃣ Si le binaire existe → OK
        if bin_path.exists():
            print(f"✔️ llama-quantize trouvé : {bin_path}")
            return str(bin_path)

        print("⚠️  llama-quantize introuvable. Compilation de llama.cpp…")

        # 2️⃣ Créer build/
        build_dir.mkdir(exist_ok=True)
        os.chdir(build_dir)

        # 3️⃣ CMake
        try:
            print("🔧 Exécution de : cmake ..")
            subprocess.check_call(["cmake", ".."])
        except subprocess.CalledProcessError as e:
            print("❌ Erreur cmake :", e)
            raise

        # 4️⃣ Compilation
        try:
            print("🔨 Compilation (make -j)…")
            subprocess.check_call(["make", "-j"])
        except subprocess.CalledProcessError as e:
            print("❌ Erreur make :", e)
            raise

        # Retour au répertoire initial
        os.chdir(str(llama_cpp_dir.parent))

        # 5️⃣ Vérification finale
        if not bin_path.exists():
            raise FileNotFoundError("❌ Compilation réussie mais llama-quantize introuvable.")

        print(f"✔️ Compilation réussie : {bin_path}")
        return str(bin_path)

    def run(self, quant_option):

        self.ensure_llama_quantize(quant_option)
        
        self.quantize_gguf(outfile)