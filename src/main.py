from script.quantizer import Quantizer
from script.pruning import Pruning
from script.llama_import import import_gguf_to_llama


def main():
    
    config_path = "etc/config.yml"

    quantizer = Quantizer(config_path)
    # pruning = Pruning(config_path) # 2 is the pruning Percentage
    quantizer.run()
    # pruning.run(2)
    #import_gguf_to_llama("gemma-3-270mprune.py")

if __name__ == "__main__":
    main()

