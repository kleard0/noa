import os
import subprocess

def create_modelfile(modelname):
	current_directory = os.getcwd()
	model_path = current_directory + "/" + modelname
	modelfile_content = f"FROM {model_path}"
	filename = "Modelfile." + modelname
	
	try:
		with open(filename, 'w') as f:
			f.write(modelfile_content)
		print(f"Fichier {filename} créé dans : {os.path.abspath(filename)}")
	except IOError as e:
		print(f"Erreur lors de la création du Modelfile : {e}")
	return filename
	
def execute_modelfile(modelname, model_file_name):
	resultat = subprocess.run(['ollama', 'create', modelname, '-f', model_file_name], check=True, capture_output=True, text=True)
	print(resultat.stdout)
	print("Your new model is now available in ollama")
	
def import_gguf_to_llama(modelname):
	model_file_name = create_modelfile(modelname)
	execute_modelfile(modelname, model_file_name)
	print(f"Modelfile pour '{modelname}' créé avec succès.")
                                                                    