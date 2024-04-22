# NOTICE: This code contains two indicies allowing remote code execution.
#         This is for debugging and compatibility reasons.
#
# [One approach to implementing https://arxiv.org/pdf/2403.13187.pdf]
#
# --Built for use in a GUI environment for quick and easy iteration.
#
# --Requires LM Evaluation Harness by EleutherAI
#   -Installing LM Eval Harness:
#       git clone https://github.com/EleutherAI/lm-evaluation-harness
#       cd lm-evaluation-harness
#       pip install -e .
#   -also requires:
#       -transformers, torch, numpy, CMA, tkinter
#
# --Limitations:
#   -It will break. It breaks assigning a selected LM Eval test as the optimization target.
# --Solutions:
#   -A heuristic to allow any test set to be ordered and prepared as an evolutionary optimization target.
# --Immediate Plans:
#   -Validate Evolutionary Merge Approach
#   -Make sure it is an accessible, rapid, and accurate approach
# --Future Plans:
#   -Command Line integration as soon as desired core functionality it matched.
#   -Several optimization methods can be applied to drastically reduce GPU/CPU/RAM usage as well as search time.
#   -Could be cannibalized into another evo-merge solution in Mergekit or diverge into a separate Mergekit method.
#       -More options are good options. 

import numpy as np
from cmaes import CMA
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from lm_eval import tasks, evaluator
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Environment variables setup for deterministic operations
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['HF_DATASETS_ALLOW_CODE_EXECUTION'] = "True"
os.environ['ALLOW_REMOTE_CODE'] = "True"

task_key_mapping = {
    "ai2_arc": "ai2_arc_challenge",
    "hellaswag": "hellaswag",
    "mmlu": "mmlu",
    "winogrande": "winogrande",
    "gsm8k": "gsm8k"
}

def select_model_folders():
    try:
        root = tk.Tk()
        root.withdraw()
        model_folder1 = os.path.abspath(filedialog.askdirectory(title="Select First Model Folder"))
        model_folder2 = os.path.abspath(filedialog.askdirectory(title="Select Second Model Folder"))
        output_folder = os.path.abspath(filedialog.askdirectory(title="Select Output Model Save Folder"))
        root.destroy()
        if not model_folder1 or not model_folder2 or not output_folder:
            raise ValueError("Model folder selection cancelled.")
        logging.info("Model folders selected successfully.")
        return [model_folder1, model_folder2, output_folder]
    except Exception as e:
        logging.error(f"Failed in select_model_folders: {e}")
        raise

def load_model(model_path):
    if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, 'config.json')):
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, low_cpu_mem_usage=False)
            logging.info(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            logging.error(f"Failed to load model from {model_path}: {e}")
            raise
    else:
        raise ValueError(f"Model path not valid or config.json not found in {model_path}")

def get_ps_config_space(num_layers):
    return [{'name': f'layer_{i}', 'type': 'real', 'bounds': [0, 1]} for i in range(num_layers * 2)]

def get_dfs_config_space(num_layers, num_models, max_path_length):
    return [{'name': f'include_layer_{i}', 'type': 'categorical', 'num_categories': 2} for i in range(num_layers * max_path_length)] + \
           [{'name': f'W_{i}_{j}', 'type': 'real', 'bounds': [0, 1]} for i in range(num_layers) for j in range(num_models)]

def evaluate(model, task_names, batch_size, limit, load_in_4bit):
    scores = {}
    with torch.no_grad():
        try:
            task_dict = tasks.get_task_dict(task_names)
            for task_name in task_names:
                if task_name in task_dict:
                    task = task_dict[task_name]
                    eval_instance = evaluator.evaluate(
                        model=model,
                        tasks=[task],
                        num_fewshot=0,
                        batch_size=batch_size,
                        limit=limit,
                        load_in_4bit=load_in_4bit,
                    )
                    scores[task_name] = eval_instance.results[task_name]
                else:
                    logging.warning(f"Task '{task_name}' not found in the task dictionary. Skipping evaluation.")
            logging.info("Evaluation completed successfully.")
        except Exception as e:
            logging.error(f"Failed in evaluate: {e}")
            raise
    return scores

def merge_ps(models, config):
    merged_model = None
    try:
        model_class = type(models[0])  # Get the actual model class
        merged_model = model_class.from_pretrained(models[0].config._name_or_path, low_cpu_mem_usage=False)  # Use the model's name or path
        num_layers = merged_model.config.num_hidden_layers
        for i in range(num_layers):
            density = config[i * 2]
            weight = config[i * 2 + 1]
            for key in models[0].state_dict().keys():
                if 'transformer.h.' in key or 'model.layers.' in key:
                    layer_index = int(key.split('.')[2])
                    if layer_index == i:
                        merged_model.state_dict()[key] = density * models[0].state_dict()[key] + (1 - density) * models[1].state_dict()[key]
        logging.info("Parameter space merging completed successfully.")
    except Exception as e:
        logging.error(f"Failed in merge_ps: {e}")
        raise
    return merged_model

def merge_dfs(models, config):
    try:
        model_config = AutoConfig.from_pretrained(models[0].config._name_or_path)
        merged_model = AutoModelForCausalLM.from_config(model_config, low_cpu_mem_usage=False)
        num_layers = models[0].config.num_hidden_layers
        num_models = len(models)
        max_path_length = num_layers * 3
        include_layers = [config[i] for i in range(num_layers * max_path_length)]
        W = [[config[num_layers * max_path_length + i * num_models + j] for j in range(num_models)] for i in range(num_layers)]
        for t in range(max_path_length):
            for m in range(num_models):
                if include_layers[t * num_models + m]:
                    layer_index = t % num_layers
                    for key in models[m].state_dict().keys():
                        if 'transformer.h.' in key or 'model.layers.' in key:
                            layer_idx = int(key.split('.')[2])
                            if layer_idx == layer_index:
                                merged_model.state_dict()[key] = models[m].state_dict()[key] * W[layer_index][m]
        logging.info("Data flow space merging completed successfully.")
        return merged_model
    except Exception as e:
        logging.error(f"Failed in merge_dfs: {e}")
        raise

def evolve_ps(models, task_names, population_size, generations, batch_size, limit, load_in_4bit):
    config_space = get_ps_config_space(models[0].config.num_hidden_layers)
    optimizer = CMA(mean=np.zeros(len(config_space)), sigma=1.0, population_size=population_size)
    best_scores = {task_name: float('-inf') for task_name in task_names}
    best_model = None
    try:
        for _ in range(generations):
            solutions = []
            for _ in range(population_size):
                config = optimizer.ask()
                merged_model = merge_ps(models, config)
                if merged_model is not None:
                    scores = evaluate(merged_model, task_names, batch_size, limit, load_in_4bit)
                    total_score = sum(scores.values())
                    solutions.append((config, total_score))
                    if all(scores[task_name] > best_scores[task_name] for task_name in task_names):
                        best_scores = scores
                        best_model = merged_model
            optimizer.tell(solutions)
        logging.info("Evolutionary search in PS completed successfully.")
    except Exception as e:
        logging.error(f"Failed in evolve_ps: {e}")
        raise
    return best_model, best_scores

def evolve_dfs(models, task_names, population_size, generations, batch_size, limit, load_in_4bit):
    config_space = get_dfs_config_space(models[0].config.num_hidden_layers, len(models), models[0].config.num_hidden_layers * 3)
    optimizer = CMA(mean=np.zeros(len(config_space)), sigma=1.0, population_size=population_size)
    best_scores = {task_name: float('-inf') for task_name in task_names}
    best_model = None
    try:
        for _ in range(generations):
            solutions = []
            for _ in range(population_size):
                config = optimizer.ask()
                merged_model = merge_dfs(models, config)
                if merged_model is not None:
                    scores = evaluate(merged_model, task_names, batch_size, limit, load_in_4bit)
                    total_score = sum(scores.values())
                    solutions.append((config, total_score))
                    if all(scores[task_name] > best_scores[task_name] for task_name in task_names):
                        best_scores = scores
                        best_model = merged_model
            optimizer.tell(solutions)
        logging.info("Evolutionary search in DFS completed successfully.")
    except Exception as e:
        logging.error(f"Failed in evolve_dfs: {e}")
        raise
    return best_model, best_scores

def merge_ps_dfs(models, task_names, ps_population_size, ps_generations, dfs_population_size, dfs_generations, batch_size, limit, load_in_4bit, output_path):
    try:
        ps_merged_model, ps_scores = evolve_ps(models, task_names, ps_population_size, ps_generations, batch_size, limit, load_in_4bit)
        if ps_merged_model is None:
            raise ValueError("PS merging failed, cannot proceed with DFS merging.")
        dfs_merged_model, dfs_scores = evolve_dfs(models + [ps_merged_model], task_names, dfs_population_size, dfs_generations, batch_size, limit, load_in_4bit)
        if dfs_merged_model:
            dfs_merged_model.save_pretrained(output_path)
            tokenizer = AutoTokenizer.from_pretrained(models[0].config._name_or_path)
            tokenizer.save_pretrained(output_path)
            logging.info(f"Final merged model and tokenizer saved to {output_path}")
        else:
            raise ValueError("DFS merging failed, model not saved.")
        return dfs_merged_model, dfs_scores
    except Exception as e:
        logging.error(f"Failed in merge_ps_dfs: {e}")
        raise

def create_task_selection_window(model_folders, output_folder):
    try:
        window = tk.Tk()
        window.title("Select Tasks To Optimize On")
        tasks_list = list(task_key_mapping.keys())
        tasks_vars = {task: tk.BooleanVar() for task in tasks_list}

        for task in tasks_list:
            tk.Checkbutton(window, text=task, variable=tasks_vars[task]).pack()

        batch_size = tk.StringVar(value="24")
        tk.Label(window, text="Batch Size:").pack()
        tk.Entry(window, textvariable=batch_size).pack()

        limit = tk.StringVar(value="300")
        tk.Label(window, text="Limit:").pack()
        tk.Entry(window, textvariable=limit).pack()

        load_in_4bit = tk.BooleanVar(value=True)
        tk.Checkbutton(window, text="Do Evaluations In 4-Bit", variable=load_in_4bit).pack()

        def on_evaluate_click():
            selected_tasks = [task for task, var in tasks_vars.items() if var.get()]
            if not selected_tasks:
                messagebox.showerror("Error", "No tasks selected. Please select at least one task.")
                return
            window.destroy()
            models = [load_model(folder) for folder in model_folders[:2]]
            mapped_tasks = [task_key_mapping[task] for task in selected_tasks]
            run_evaluation(models, mapped_tasks, int(batch_size.get()), int(limit.get()), load_in_4bit.get(), output_folder)

        tk.Button(window, text="Evolve", command=on_evaluate_click).pack()
        window.mainloop()
        logging.info("Task selection window displayed successfully.")
    except Exception as e:
        logging.error(f"Failed in create_task_selection_window: {e}")
        raise

def run_evaluation(models, mapped_tasks, batch_size, limit, load_in_4bit, output_path):
    try:
        merged_model, scores = merge_ps_dfs(models, mapped_tasks, ps_population_size=50, ps_generations=100, dfs_population_size=50, dfs_generations=100, batch_size=batch_size, limit=limit, load_in_4bit=load_in_4bit, output_path=output_path)
        logging.info(f"Best merged model saved to '{output_path}'. Scores: {scores}")
    except Exception as e:
        logging.error(f"Failed in run_evaluation: {e}")
        raise

def main():
    try:
        model_folders = select_model_folders()
        if len(model_folders) == 3:
            output_folder = model_folders[2]
            create_task_selection_window(model_folders, output_folder)
        else:
            raise ValueError("Incorrect folder selection. Please select exactly two model folders and one output folder.")
    except Exception as e:
        logging.error(f"Failed in main: {e}")
        raise

if __name__ == "__main__":
    main()