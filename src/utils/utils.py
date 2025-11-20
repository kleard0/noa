import yaml
import os
from huggingface_hub import snapshot_download, login
from pathlib import Path



def load_config(config_path):
    print(config_path)
    """Charge les paramètres de configuration à partir d'un fichier YAML."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Fichier de configuration non trouvé: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    print("## Configuration Chargée ##")
    for key, value in config.items():
        print(f"  > {key}: {value}")
    
    return config


def download_hf_model(self):
    """
    Methods to download a model (llm) from hf, login() method use HF_TOKEN
    """
    try : 
        login(self.hf_token)
        snapshot_download(repo_id=self.model_path, local_dir=self.model_name)
    except Exception as e:
        print(f"Erreur lors du téléchargement : {e}")
        raise


def check_model_folder(self):
    hf_dir = Path(self.model_name)
    if not hf_dir.exists():
        raise SystemExit(f"Répertoire {hf_dir} introuvable. Vérifie snapshot_download.")
    print(f"Contenu de {hf_dir}:")
    
    for p in sorted(hf_dir.rglob("*")):
        rel = p.relative_to(hf_dir)
        print(rel)
    expected = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    weights = list(hf_dir.rglob("*.safetensors")) + list(hf_dir.rglob("pytorch_model*.bin"))
    print("\nFichiers de config/tokenizer attendus présents:")
    for e in expected:
        print(f" - {e} :", (hf_dir / e).exists())
    print("\nFichiers de poids trouvés:")
    for w in weights:
        print(" -", w.relative_to(hf_dir))
    if not weights:
        print(" -> Aucun fichier de poids (.safetensors / pytorch_model.bin) trouvé. Vérifie le repo HF.")
    for p in Path("llama.cpp").rglob("convert*.py"):
        print(p)