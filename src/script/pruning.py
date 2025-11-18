from huggingface_hub import snapshot_download, login, notebook_login
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
import subprocess
import yaml
import torch
import shutil
from torch.nn.utils import prune
from utils.utils import load_config
from pathlib import Path


class Pruning():
	def __init__(self, config_path):

		self.config = load_config(config_path)
		self.model_path = self.config["model_path"]
		self.model_name = self.config["model_name"]
		os.environ['HF_TOKEN'] = self.config["HF_token"]



	def convert_hf_to_gguf(self):

		model_dir = self.model_name
		outfile = self.model_file_name

		cmd = [
			sys.executable,
			"llama.cpp/convert_hf_to_gguf.py",
			model_dir,
			"--outfile", outfile,
			"--outtype", "F32"
		]
		print("🚀 Conversion HF → GGUF")
		print(">>>", " ".join(cmd))

		try:
			subprocess.check_call(cmd)
			print(f"✔️ GGUF généré : {outfile}")
		except subprocess.CalledProcessError as e:
			print("❌ Erreur lors de la conversion :", e)
			raise
		return outfile
	
	def charge_model(self):
		try:
			HF_DIR = Path(self.model_name)
			model = AutoModelForCausalLM.from_pretrained(
				HF_DIR, 
				dtype=torch.float16, 
				device_map="auto"
			)
			tokenizer = AutoTokenizer.from_pretrained(HF_DIR)
			all_initial_files = list(HF_DIR.rglob("*.safetensors")) + list(HF_DIR.rglob("pytorch_model*.bin"))
			initial_size = sum(os.path.getsize(f) for f in all_initial_files)
			print(f"Modèle chargé. Taille initiale: {initial_size / (1024*1024):.2f} MB")
		except Exception as e:
			raise SystemExit(f"Erreur lors du chargement: {e}")

		return model, tokenizer	
		
	def prune_safetensor(self, pruningPercentage):
		model, tokenizer = self.charge_model()
		PRUNING_AMOUNT = pruningPercentage
		modules_to_prune = []
		for name, module in model.named_modules():
			if isinstance(module, torch.nn.Linear) and "embed" not in name:
				modules_to_prune.append((module, 'weight'))

		# Application du Pruning Non Structuré (L1 basé sur la magnitude)
		for module, name in modules_to_prune:
			prune.l1_unstructured(module, name=name, amount=PRUNING_AMOUNT)

		for module, name in modules_to_prune:
			prune.remove(module, name)
		
		self.save_model(model, tokenizer)
	
	def save_model(self, model, tokenizer):
		PRUNED_DIR = Path(self.model_name)

		model.save_pretrained(PRUNED_DIR)
		tokenizer.save_pretrained(PRUNED_DIR)
	
	def run(self, pruningPercentage):


		self.prune_safetensor(pruningPercentage)		
		print("Convertion en gguf")
		outfile = self.convert_hf_to_gguf()