import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.cross_decomposition import CCA
from sklearn.manifold import SpectralEmbedding, LocallyLinearEmbedding, Isomap
from scipy.linalg import orthogonal_procrustes
from sklearn.preprocessing import normalize
import tkinter as tk
from tkinter import filedialog
import yaml
import sys
import os

os.environ['USE_FLASH_ATTENTION'] = "1"
sys.stdout.flush()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def select_model_folder(title="Select Model Folder"):
    root = tk.Tk()
    root.withdraw()
    return filedialog.askdirectory(title=title)

print("Select the Reference Model:")
base_model_folder = select_model_folder("Select Reference Model")
base_model_tokenizer = AutoTokenizer.from_pretrained(base_model_folder)

num_models = int(input("Enter the number of models to extract elements from: "))
models = []
for i in range(num_models):
    print(f"Select folder for Model {i+1}:")
    models.append(select_model_folder())

alignment_methods = {
    1: "Manifold: Spectral Embedding",
    2: "Manifold: Locally Linear Embedding",
    3: "Isomap",
    4: "Orthogonal Procrustes",
    5: "Canonical Correlation Analysis (CCA)"
}
print("Select the alignment method:")
for key, value in alignment_methods.items():
    print(f"{key}: {value}")

while True:
    try:
        selected_method = int(input("Enter the number of the alignment method: "))
        if selected_method in alignment_methods:
            break
        else:
            print("Invalid choice. Please select a number from the available options.")
    except ValueError:
        print("Invalid input. Please enter a valid integer corresponding to the alignment methods.")

with open("phrase_dictionary.txt", "r") as file:
    phrases = file.read().strip().split('\n')

def extract_embeddings(model_folder, tokenizer, phrases, max_length=64, batch_size=256):
    model = AutoModelForCausalLM.from_pretrained(model_folder, load_in_4bit=True, device_map="auto")
    tokenizer.pad_token = tokenizer.eos_token
    embeddings_list = []
    num_batches = (len(phrases) + batch_size - 1) // batch_size
    for i in range(0, len(phrases), batch_size):
        batch_phrases = phrases[i:i + batch_size]
        inputs = tokenizer(batch_phrases, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            embeddings = outputs.hidden_states[-1].mean(dim=1)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            embeddings_list.append(embeddings.cpu().to(dtype=torch.float32))  # Move embeddings to CPU and convert to float32
            for j, phrase in enumerate(batch_phrases):
                embedding_num = i + j + 1
                print(f"Extracting embedding {embedding_num}/{len(phrases)} from model: {model_folder}")
                print(f"Phrase: {phrase}")
                print("---")
    del model
    torch.cuda.empty_cache()
    return torch.stack(embeddings_list, dim=0)

def preprocess_embeddings(embeddings):
    embeddings = embeddings.view(-1, embeddings.size(-1)) if embeddings.dim() > 2 else embeddings
    embeddings = embeddings.numpy()
    embeddings = normalize(embeddings, axis=1)
    return torch.from_numpy(embeddings).to(dtype=torch.float32)  # Convert back to tensor and ensure float32 dtype

base_embeddings = preprocess_embeddings(extract_embeddings(base_model_folder, base_model_tokenizer, phrases))
embeddings_dict = {base_model_folder: base_embeddings}
for model_folder in models:
    tokenizer = AutoTokenizer.from_pretrained(model_folder)
    embeddings = preprocess_embeddings(extract_embeddings(model_folder, tokenizer, phrases))
    embeddings_dict[model_folder] = embeddings

def calculate_n_components(embeddings1, embeddings2):
    return min(embeddings1.shape[0], embeddings1.shape[1], embeddings2.shape[0], embeddings2.shape[1])

def align_embeddings_manifold(embeddings1, embeddings2, method):
    if method == "spectral":
        alignment = SpectralEmbedding(n_components=calculate_n_components(embeddings1, embeddings2))
    elif method == "lle":
        alignment = LocallyLinearEmbedding(n_components=calculate_n_components(embeddings1, embeddings2))
    elif method == "isomap":
        alignment = Isomap(n_components=calculate_n_components(embeddings1, embeddings2))
    stacked_embeddings = np.vstack([embeddings1.numpy(), embeddings2.numpy()])
    aligned_embeddings = alignment.fit_transform(stacked_embeddings)
    split_index = embeddings1.shape[0]
    return torch.from_numpy(aligned_embeddings[:split_index]), torch.from_numpy(aligned_embeddings[split_index:])

def align_embeddings_procrustes(embeddings1, embeddings2):
    R, _ = orthogonal_procrustes(embeddings2.numpy(), embeddings1.numpy())
    aligned_embeddings2 = torch.matmul(embeddings2, torch.from_numpy(R))
    return embeddings1, aligned_embeddings2

# 2000 iter and 1e-5 to tighten alignment more
def align_embeddings_cca(embeddings1, embeddings2, max_iter=500, tol=1e-3):
    # Reduced the number of iterations and increased the tolerance
    cca = CCA(n_components=min(embeddings1.shape[1], embeddings2.shape[1]), max_iter=max_iter, tol=tol)
    cca.fit(embeddings1.numpy(), embeddings2.numpy())
    aligned_embeddings1, aligned_embeddings2 = cca.transform(embeddings1.numpy(), embeddings2.numpy())
    return torch.from_numpy(aligned_embeddings1), torch.from_numpy(aligned_embeddings2)

alignment_functions = {
    1: lambda e1, e2: align_embeddings_manifold(e1, e2, "spectral"),
    2: lambda e1, e2: align_embeddings_manifold(e1, e2, "lle"),
    3: lambda e1, e2: align_embeddings_manifold(e1, e2, "isomap"),
    4: align_embeddings_procrustes,
    5: align_embeddings_cca
}

align_func = alignment_functions[selected_method]
all_aligned_embeddings = []
for model_folder, embeddings in embeddings_dict.items():
    if model_folder != base_model_folder:
        _, aligned_embeddings = align_func(base_embeddings, embeddings)
        all_aligned_embeddings.append(aligned_embeddings)

# Postprocess to ensure all embeddings are tensors of the same dtype
def postprocess_embeddings(embeddings_list):
    dtype = torch.float32
    for emb in embeddings_list:
        if isinstance(emb, torch.Tensor):
            dtype = emb.dtype
            break

    processed_embeddings = []
    for emb in embeddings_list:
        if not isinstance(emb, torch.Tensor):
            emb = torch.from_numpy(emb)
        processed_embeddings.append(emb.to(dtype=dtype))

    return processed_embeddings

all_aligned_embeddings = postprocess_embeddings(all_aligned_embeddings)
merged_embeddings = torch.mean(torch.stack(all_aligned_embeddings), dim=0)

# Re-align the merged embeddings with the base model's embeddings
_, final_aligned_embeddings = align_func(base_embeddings, merged_embeddings)

# Postprocess the final aligned embeddings
final_aligned_embeddings = postprocess_embeddings([final_aligned_embeddings])[0]

# Merge the final aligned embeddings into the last three layers of the new base model
new_base_model = AutoModelForCausalLM.from_pretrained(base_model_folder).cpu()

# Get the layers of the base model
base_model_layers = new_base_model.model.layers

# Get the last three layers
layers_to_merge = base_model_layers[-5:]

# Define the gradient of influence for each layer
gradient_weights = [0.05, 0.09, 0.16, 0.28, 0.5]  # Adjust these weights as needed

# Loop over the layers and merge the final aligned embeddings into the parameters
for layer, weight in zip(layers_to_merge, gradient_weights):
    for name, param in layer.named_parameters():
        param_size = param.data.numel()  # Get the total number of elements in param.data

        # Calculate required number of rows to be repeated
        rows_needed = (param_size + final_aligned_embeddings.shape[1] - 1) // final_aligned_embeddings.shape[1]
        
        # Check if repetition of rows is needed
        if rows_needed > final_aligned_embeddings.shape[0]:
            row_repeats = (rows_needed + final_aligned_embeddings.shape[0] - 1) // final_aligned_embeddings.shape[0]
            extended_embeddings = final_aligned_embeddings.repeat(row_repeats, 1)
        else:
            extended_embeddings = final_aligned_embeddings

        # Slice to match the size and reshape
        extended_embeddings = extended_embeddings.view(-1)[:param_size]
        sliced_embeddings = extended_embeddings.view(param.data.shape)

        # Compute the weighted combination of original and new weights
        merged_weights = (1 - weight) * param.data + weight * sliced_embeddings
        param.data.copy_(merged_weights)

output_dir = "aligned_model"
new_base_model.save_pretrained(output_dir)
print(f"Aligned model saved successfully to {output_dir}")
# Save the tokenizer with the model
new_base_model_tokenizer.save_pretrained(output_dir)
print(f"Tokenizer saved successfully to {output_dir}")
