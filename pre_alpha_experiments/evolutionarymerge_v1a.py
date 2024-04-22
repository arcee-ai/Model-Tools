import numpy as np
import pandas as pd
from cmaes import CMA
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from lm_eval import tasks, evaluator
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import tempfile
import shutil
import torch
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Any
import yaml

# loads models before and after options menu, will remove that redundancy - easy
# fails, stating there is no attribute num_layers - should be an easy fix to pull
# from the model's config and sanity check that is passed where it needs to go.
# all calcs done based on num_layers of one model wherether it be PS or DFS,
# math is done to scale what is needed to properly calculate PS or DFS
# based on num_layers(it's basically the pinion of genome size calc)

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Setup environment variables for deterministic operations
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['HF_DATASETS_ALLOW_CODE_EXECUTION'] = "True"
os.environ['ALLOW_REMOTE_CODE'] = "True"

class EvolMergeConfiguration:
    def __init__(self, tasks):
        self.tasks = tasks

class ModelConfig:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        self.num_layers = self.model.config.num_hidden_layers

    @staticmethod
    def load_model(model_path):
        return AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)

def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model

def genotype_merge_config(models):
    num_layers = max(model.num_layers for model in models)
    num_models = len(models)

    ps_genotype_length = num_layers * 2
    dfs_genotype_length = num_layers * num_layers * num_models
    expected_genotype_length = ps_genotype_length + dfs_genotype_length

    logging.debug(f"Expected genotype length: {expected_genotype_length}")
    return expected_genotype_length, ps_genotype_length, dfs_genotype_length

def initialize_genotype(length):
    return np.random.rand(length)

def select_model_folders():
    root = tk.Tk()
    root.withdraw()
    model_folder1 = filedialog.askdirectory(title="Select First Model Folder")
    model_folder2 = filedialog.askdirectory(title="Select Second Model Folder")
    output_folder = filedialog.askdirectory(title="Select Output Model Save Folder")
    root.destroy()

    if not model_folder1 or not model_folder2 or not output_folder:
        raise ValueError("Model folder selection cancelled.")
    logging.info("Model folders selected successfully.")
    return [model_folder1, model_folder2, output_folder]

## I added in get_ps_config_space and get_dfs_config_space from evolution4_claude_neo5b.py
## this revision just.. doesn't have these... did they even get used in any previous scripts
## or did they just sit here for no reason because Claude was convinced this did something?
def get_ps_config_space(num_layers):
    # Parameter Space: Each layer has two parameters (e.g., mix ratio and activation function choice)
    return [{'name': f'layer_{i}', 'type': 'real', 'bounds': [0, 1]} for i in range(num_layers * 2)]

def get_dfs_config_space(num_layers, num_models, max_path_length):
    # Data Flow Space: Control the flow and mixture of data through layers and across models
    return [{'name': f'include_layer_{i}', 'type': 'categorical', 'num_categories': 2} for i in range(num_layers * max_path_length)] + \
           [{'name': f'W_{i}_{j}', 'type': 'real', 'bounds': [0, 1]} for i in range(num_layers) for j in range(num_models)]

def evaluate_model(model, tasks, model_args, task_manager, **kwargs):
    results = evaluator.simple_evaluate(
        model=model,
        model_args=model_args,
        tasks=tasks,
        log_samples=False,
        verbosity="WARNING",
        task_manager=task_manager,
        **kwargs,
    )
    logging.info(results["results"])
    return {"score": sum(results["results"].values()), "results": results["results"]}

def merge_model(genotype, models, model_storage_path):
    expected_len, ps_len, dfs_len = genotype_merge_config(models)
    if len(genotype) != expected_len:
        raise ValueError(f"Genotype length mismatch: Expected {expected_len}, got {len(genotype)}")

    ps_genotype = genotype[:ps_len]
    dfs_genotype = genotype[ps_len:ps_len+dfs_len]
    os.makedirs(model_storage_path, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="merged-", dir=model_storage_path)

    # Assuming all models are of the same class
    merged_model = models[0].__class__.from_pretrained(models[0].model.config._name_or_path, low_cpu_mem_usage=False)
    num_layers = merged_model.config.num_hidden_layers

    # Merge in the Parameter Space (PS)
    for i in range(num_layers):
        density = ps_genotype[i * 2]
        weight = ps_genotype[i * 2 + 1]
        for key in models[0].model.state_dict().keys():
            if 'model' in key:
                merged_model.state_dict()[key] = (density * models[0].model.state_dict()[key] + 
                                                  (1 - density) * models[1].model.state_dict()[key])

    # Merge in the Data Flow Space (DFS)
    for t in range(num_layers * 3):  # Assuming `num_layers * 3` path length adjustments
        for m in range(len(models)):
            if dfs_genotype[t * len(models) + m] > 0.5:
                layer_index = t % num_layers
                for key in models[m].model.state_dict().keys():
                    if 'model' in key:
                        merged_model.state_dict()[key] *= dfs_genotype[-num_layers * num_models + layer_index * num_models + m]

    merged_model.save_pretrained(temp_dir)
    logging.info("Model merging completed successfully.")
    return temp_dir

def create_task_selection_window(model_folders, output_folder):
    window = tk.Tk()
    window.title("Select Tasks To Optimize On")
    with open("test_options.yaml", 'r') as file:
        tasks_config = yaml.safe_load(file)
        tasks_list = tasks_config.get('tests', [])
    tasks_vars = {task: tk.BooleanVar(value=False) for task in tasks_list}
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
    ps_population_size = tk.StringVar(value="50")
    tk.Label(window, text="PS Population Size:").pack()
    tk.Entry(window, textvariable=ps_population_size).pack()
    ps_generations = tk.StringVar(value="100")
    tk.Label(window, text="PS Generations:").pack()
    tk.Entry(window, textvariable=ps_generations).pack()
    dfs_population_size = tk.StringVar(value="50")
    tk.Label(window, text="DFS Population Size:").pack()
    tk.Entry(window, textvariable=dfs_population_size).pack()
    dfs_generations = tk.StringVar(value="100")
    tk.Label(window, text="DFS Generations:").pack()
    tk.Entry(window, textvariable=dfs_generations).pack()
    def on_evaluate_click():
        selected_tasks = [task for task, var in tasks_vars.items() if var.get()]
        if not selected_tasks:
            messagebox.showerror("Error", "No tasks selected. Please select at least one task.")
            return
        window.destroy()
        models = [load_model(folder) for folder in model_folders[:2]]
        run_evaluation(models, selected_tasks, int(batch_size.get()), int(limit.get()), load_in_4bit.get(), output_folder,
                       int(ps_population_size.get()), int(ps_generations.get()), int(dfs_population_size.get()), int(dfs_generations.get()))
    tk.Button(window, text="Evolve", command=on_evaluate_click).pack()
    window.mainloop()

class EvaluationStrategyBase(ABC):
    def __init__(self, config, model_storage_path, num_gpus=None, batch_size=None, task_search_path=None):
        self.config = config
        self.model_storage_path = model_storage_path
        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.batch_size = batch_size
        self.task_manager = tasks.TaskManager(include_path=task_search_path)

    @abstractmethod
    def evaluate_genotypes(self, genotypes, models):
        pass

    @abstractmethod
    def evaluate_genotype(self, genotype, models):
        pass

class SerialEvaluationStrategy(EvaluationStrategyBase):
    def evaluate_genotypes(self, genotypes, models):
        scores = []
        for genotype in genotypes:
            model_path = merge_model(genotype, models, self.model_storage_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)
            score = evaluate_model(model, self.config.tasks, {}, None)["score"]
            scores.append(score)
            shutil.rmtree(model_path)
        return scores

    def evaluate_genotype(self, genotype, models):
        return self.evaluate_genotypes([genotype], models)[0]

def merge_ps_dfs(models, task_names, ps_population_size, ps_generations, dfs_population_size, dfs_generations, batch_size, limit, output_path):
    try:
        # Initialize the evaluation strategy with the model configuration and task settings
        evaluation_strategy = SerialEvaluationStrategy(
            config=EvolMergeConfiguration(tasks=task_names),
            model_storage_path=output_path,
            batch_size=batch_size,
        )

        # Evolutionary strategy for Parameter Space
        ps_best_model, ps_scores = evolve_ps(
            models, task_names, ps_population_size, ps_generations, batch_size, limit, evaluation_strategy
        )
        
        if ps_best_model is None:
            raise ValueError("PS merging failed, cannot proceed with DFS merging.")

        # Loading the best PS model to include in DFS evolution
        ps_best_model_config = ModelConfig(ps_best_model)
        models.append(ps_best_model_config)

        # Evolutionary strategy for Data Flow Space
        dfs_best_model, dfs_scores = evolve_dfs(
            models, task_names, dfs_population_size, dfs_generations, batch_size, limit, evaluation_strategy
        )
        
        if dfs_best_model is None:
            raise ValueError("DFS merging failed, model not saved.")
        
        # Saving the best DFS model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(models[0].model.config._name_or_path)
        dfs_best_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        logging.info(f"Final merged model and tokenizer saved to {output_path}")
        return dfs_best_model, dfs_scores

    except Exception as e:
        logging.error(f"Failed in merge_ps_dfs: {e}")
        raise

# Supporting functions for PS and DFS evolution
def evolve_ps(models, task_names, population_size, generations, batch_size, limit, evaluation_strategy):
    num_layers = max(model.num_layers for model in models)
    ps_genotype_length = num_layers * 2
    optimizer = CMA(mean=np.zeros(ps_genotype_length), sigma=1.0, population_size=population_size)
    best_scores = {task_name: float('-inf') for task_name in task_names}
    best_model = None

    for _ in range(generations):
        genotypes = [optimizer.ask() for _ in range(population_size)]
        scores = evaluation_strategy.evaluate_genotypes(genotypes, models)
        optimizer.tell(list(zip(genotypes, scores)))
        for score, genotype in zip(scores, genotypes):
            if all(score > best_scores[task_name] for task_name in task_names):
                best_scores = {task_name: score for task_name in task_names}
                best_model = genotype  # Storing the best genotype

    return best_model, best_scores

def evolve_dfs(models, task_names, population_size, generations, batch_size, limit, evaluation_strategy):
    num_layers = max(model.num_layers for model in models)
    num_models = len(models)
    max_path_length = num_layers * 3  # Simplified path length estimation
    dfs_genotype_length = num_layers * max_path_length + num_layers * num_models
    optimizer = CMA(mean=np.zeros(dfs_genotype_length), sigma=1.0, population_size=population_size)
    best_scores = {task_name: float('-inf') for task_name in task_names}
    best_model = None

    for _ in range(generations):
        genotypes = [optimizer.ask() for _ in range(population_size)]
        scores = evaluation_strategy.evaluate_genotypes(genotypes, models)
        optimizer.tell(list(zip(genotypes, scores)))
        for score, genotype in zip(scores, genotypes):
            if all(score > best_scores[task_name] for task_name in task_names):
                best_scores = {task_name: score for task_name in task_names}
                best_model = genotype  # Storing the best genotype

    return best_model, best_scores

def run_evaluation(models, selected_tasks, batch_size, limit, load_in_4bit, output_path,
                   ps_population_size, ps_generations, dfs_population_size, dfs_generations):
    try:
        merged_model, scores = merge_ps_dfs(models, selected_tasks, ps_population_size, ps_generations,
                                            dfs_population_size, dfs_generations, batch_size, limit, output_path)
        logging.info(f"Best merged model saved to '{output_path}'. Scores: {scores}")
        df_scores = pd.DataFrame.from_dict(scores, orient='index', columns=['Score'])
        print("Scores DataFrame:")
        print(df_scores)
    except Exception as e:
        logging.error(f"Failed in run_evaluation: {e}")
        raise

def main():
    try:
        model_folders = select_model_folders()
        output_folder = model_folders.pop()
        models = [ModelConfig(path) for path in model_folders]
        create_task_selection_window(model_folders, output_folder)
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
