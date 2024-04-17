# LEARNMORE is Latent Embedding Alignment and Reconstruction for Neural Model Optimization, Refinement, and Enhancement
# [the acronym isn't perfect yet, the name is not quite the priority]
#
# --Functional Overview:
# -Extracts, aligns, and merges embeddings from an arbitrary amount of models into a base model.
# -Uses Scaled Dot Product optimization 
# -User selects arbitrary num models above 1
# -Model select popup for each model
# -User provides phrase dictionary phrase_dictionary.yaml
#   -each entry is individually tokenized, sent through every model, and embedding spaces are extracted from models
# -Embedding alignment process begins
#   -five alignment options to organize embeddings
#       -Orthogonal Pro-crus-tes
#       -Canonical Correlation Analysis (CCA)
#       -Manifold Alignment:
#           -Locally Linear
#           -Spectral
#           -Isomap
# -Embeddings are aligned and injected into base model
#   -first selected model is the base (for now)
# -Embedding augmented model is saved to output folder (hardcoded for now)
#
# --Limitations:
#   -Will likely break, working beta version is on the horizon.
#
# --Requirements:
#   -a GUI environment (for now; in place for quick testing and iteration)
#   -transformers, torch, sklearn, scipy, tkinter, yaml
#
# --Future Plans:
#   -An impeccable amount of testing.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.cross_decomposition import CCA
from sklearn.manifold import SpectralEmbedding, LocallyLinearEmbedding, Isomap
from scipy.linalg import orthogonal_procrustes
import tkinter as tk
from tkinter import filedialog
import yaml

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def select_model_folder():
    root = tk.Tk()
    root.withdraw()
    model_folder = filedialog.askdirectory(title="Select Model Folder")
    return model_folder

# Step 1: User selects the number of models to merge
num_models = int(input("Enter the number of models you want to merge: "))

models = []
for i in range(num_models):
    print(f"Select folder for Model {i+1}:")
    model_folder = select_model_folder()
    models.append(model_folder)

# Load the phrase dictionary from an external file
with open("phrase_dictionary.yaml", "r") as file:
    phrase_dictionary = yaml.safe_load(file)

# Step 2: Extract embedding spaces
def extract_embeddings(model_folder, tokenizer, layer_idx, phrase_dictionary, max_length=512):
    model = AutoModelForCausalLM.from_pretrained(model_folder, device_map="auto")
    
    # Tokenize the phrases from the phrase dictionary
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(phrase_dictionary, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        embeddings = outputs.hidden_states[layer_idx]
        embeddings = embeddings.mean(dim=1)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # Normalize embeddings
        embeddings = embeddings.cpu()  # Move embeddings back to CPU for compatibility
    
    del model  # Unload the model from GPU memory
    torch.cuda.empty_cache()  # Clear GPU cache
    
    return embeddings

# Extract embeddings from all layers
embeddings_dict = {}
for i, model_folder in enumerate(models):
    model_name = f"model{i+1}"
    print(f"Extracting embeddings from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_folder)
    model = AutoModelForCausalLM.from_pretrained(model_folder, device_map="auto")
    num_layers = model.config.num_hidden_layers
    embeddings_list = []
    for layer_idx in range(num_layers):
        print(f"Extracting embeddings from {model_name}, Layer {layer_idx+1}/{num_layers}")
        embeddings = extract_embeddings(model_folder, tokenizer, layer_idx, phrase_dictionary)
        embeddings_list.append(embeddings)
    embeddings_dict[model_name] = embeddings_list
    print(f"Finished extracting embeddings from {model_name}")

# Step 3: Multiple alignment methods available for user selection
def align_embeddings_manifold(embeddings_list1, embeddings_list2, method="spectral"):
    embeddings_concat1 = torch.cat(embeddings_list1, dim=0)
    embeddings_concat2 = torch.cat(embeddings_list2, dim=0)
    
    if method == "spectral":
        manifold_alignment = SpectralEmbedding(n_components=embeddings_list1[0].shape[1])
    elif method == "lle":
        manifold_alignment = LocallyLinearEmbedding(n_components=embeddings_list1[0].shape[1])
    elif method == "isomap":
        manifold_alignment = Isomap(n_components=embeddings_list1[0].shape[1])
    else:
        raise ValueError(f"Invalid manifold alignment method: {method}")
    
    aligned_embeddings_concat = manifold_alignment.fit_transform(torch.cat((embeddings_concat1, embeddings_concat2), dim=0))
    aligned_embeddings_list1 = torch.split(torch.tensor(aligned_embeddings_concat[:embeddings_concat1.shape[0]]), embeddings_list1[0].shape[0])
    aligned_embeddings_list2 = torch.split(torch.tensor(aligned_embeddings_concat[embeddings_concat1.shape[0]:]), embeddings_list2[0].shape[0])
    
    return aligned_embeddings_list1, aligned_embeddings_list2

def align_embeddings_procrustes(embeddings_list1, embeddings_list2):
    embeddings_concat1 = torch.cat(embeddings_list1, dim=0)
    embeddings_concat2 = torch.cat(embeddings_list2, dim=0)
    
    R, _ = orthogonal_procrustes(embeddings_concat1.numpy(), embeddings_concat2.numpy())
    R = torch.from_numpy(R)
    aligned_embeddings_concat1 = torch.matmul(embeddings_concat1, R)
    
    aligned_embeddings_list1 = torch.split(aligned_embeddings_concat1, embeddings_list1[0].shape[0])
    aligned_embeddings_list2 = embeddings_list2
    
    return aligned_embeddings_list1, aligned_embeddings_list2

def align_embeddings_cca(embeddings_list1, embeddings_list2):
    cca = CCA(n_components=min(embeddings_list1[0].shape[1], embeddings_list2[0].shape[1]))
    cca.fit(embeddings_list1[0], embeddings_list2[0])
    aligned_embeddings_list1 = [cca.transform(embeddings) for embeddings in embeddings_list1]
    aligned_embeddings_list2 = [cca.transform(embeddings) for embeddings in embeddings_list2]
    return aligned_embeddings_list1, aligned_embeddings_list2

print("Available alignment methods:")
print("1. Manifold Alignment (Spectral Embedding)")
print("2. Manifold Alignment (Locally Linear Embedding)")
print("3. Manifold Alignment (Isomap)")
print("4. Orthogonal Procrustes")
print("5. Canonical Correlation Analysis (CCA)")
while True:
    try:
        alignment_method = int(input("Enter the alignment method number: ").strip())
        if alignment_method in [1, 2, 3, 4, 5]:
            break  # If the input is valid and within the allowed range, break out of the loop
        else:
            print("Invalid choice. Please select a number between 1 and 5.")
    except ValueError:
        print("Invalid input. Please enter a valid integer corresponding to the alignment methods.")

# Step 4: Align representations

def safe_model_to_cpu(model):
    for name, param in model.named_parameters():
        if 'meta' not in str(param.device):
            param.data = param.data.to('cpu')  # Safely move each parameter to CPU if it's not offloaded
    return model
def align_models(embeddings_dict, alignment_method):
    if alignment_method == 1:
        align_func = lambda x, y: align_embeddings_manifold(x, y, method="spectral")
    elif alignment_method == 2:
        align_func = lambda x, y: align_embeddings_manifold(x, y, method="lle")
    elif alignment_method == 3:
        align_func = lambda x, y: align_embeddings_manifold(x, y, method="isomap")
    elif alignment_method == 4:
        align_func = align_embeddings_procrustes
    elif alignment_method == 5:
        align_func = align_embeddings_cca
    else:
        raise ValueError(f"Invalid alignment method: {alignment_method}")
    
    model_names = list(embeddings_dict.keys())
    reference_model_name = model_names[0]
    aligned_model = AutoModelForCausalLM.from_pretrained(models[0], device_map="auto")
    
    for model_name in model_names[1:]:
        print(f"Aligning embeddings between {reference_model_name} and {model_name}")
        embeddings_list1, embeddings_list2 = embeddings_dict[reference_model_name], embeddings_dict[model_name]
        aligned_embeddings1, aligned_embeddings2 = align_func(embeddings_list1, embeddings_list2)
        aligned_model.get_input_embeddings().weight.data = aligned_embeddings1[-1].to(aligned_model.device)
    
    aligned_model = safe_model_to_cpu(aligned_model)  # Move the model safely to CPU
    return aligned_model

aligned_model = align_models(embeddings_dict, alignment_method)

# Save the aligned model
output_dir = "aligned_model"
aligned_model.save_pretrained(output_dir)
print(f"Aligned model saved successfully to {output_dir}")