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

# genotype_merge_config has expected_genotype_length hardcoded to 64 for debugging other areas of the code;
# expected_genotype_length consistently calculates an integer > 3000 yet sees 64 when everything is orchestrated
# and it creates a mismatch error. Code will run for some time, maxing CPU and RAM then return 
# IndexError: index 0 is out of bounds for axis 0 with size 0 which cause 90% likely the genome hardcoding
# Also commented out expected size vs loaded and calced size sanity check for debug.

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Environment variables setup for deterministic operations
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['HF_DATASETS_ALLOW_CODE_EXECUTION'] = "True"
os.environ['ALLOW_REMOTE_CODE'] = "True"

class EvolMergeConfiguration:
    def __init__(self, tasks):
        self.tasks = tasks

def genotype_merge_config(genotype, models):
    num_layers = max(model.config.num_hidden_layers for model in models)  # Consider the maximum to accommodate all models
    num_models = len(models)
    max_path_length = num_layers * 3  # Dynamic based on the deepest model

    ps_genotype_length = 64 #num_layers * 2
    dfs_genotype_length = 128 #num_layers * max_path_length + num_layers * num_models
    expected_genotype_length = ps_genotype_length + dfs_genotype_length

    logging.debug(f"Genotype length: {len(genotype)}, Expected: {expected_genotype_length}")

#    if len(genotype) != expected_genotype_length:
#        logging.error(f"Genotype length mismatch: Expected {expected_genotype_length}, got {len(genotype)}")
#        raise ValueError(f"Genotype length mismatch: Expected {expected_genotype_length}, got {len(genotype)}")

    # Split genotype into respective configurations
    split_at_ps = ps_genotype_length
    split_at_dfs = split_at_ps + num_layers * max_path_length
    ps_config = genotype[:split_at_ps]
    dfs_config = genotype[split_at_ps:split_at_dfs]
    W_config = genotype[split_at_dfs:]

    cfg = {
        'ps': ps_config,
        'dfs': {
            'include_layers': dfs_config,
            'W': np.reshape(W_config, (num_layers, num_models)) if W_config.size else np.zeros((num_layers, num_models))
        }
    }

    return cfg, ps_genotype_length, dfs_genotype_length

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
    # Parameter Space: Each layer has two parameters (e.g., mix ratio and activation function choice)
    return [{'name': f'layer_{i}', 'type': 'real', 'bounds': [0, 1]} for i in range(num_layers * 2)]

def get_dfs_config_space(num_layers, num_models, max_path_length):
    # Data Flow Space: Control the flow and mixture of data through layers and across models
    return [{'name': f'include_layer_{i}', 'type': 'categorical', 'num_categories': 2} for i in range(num_layers * max_path_length)] + \
           [{'name': f'W_{i}_{j}', 'type': 'real', 'bounds': [0, 1]} for i in range(num_layers) for j in range(num_models)]

def evaluate_model(
    model: Union[str, AutoModelForCausalLM],
    tasks: List[str],
    model_args: Optional[Dict[str, Any]] = None,
    task_manager: Optional[tasks.TaskManager] = None,
    **kwargs,
) -> Dict[str, Any]:
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

def merge_model(genotype: np.ndarray, models, model_storage_path: str):
    cfg, _, _ = genotype_merge_config(genotype, models)
    os.makedirs(model_storage_path, exist_ok=True)
    res = tempfile.mkdtemp(prefix="merged", dir=model_storage_path)
    
    # Perform model merging
    model_class = type(models[0])  # Assuming all models are of the same class
    merged_model = model_class.from_pretrained(models[0].config._name_or_path, low_cpu_mem_usage=False)
    num_layers = merged_model.config.num_hidden_layers
    
    # Merge in the Parameter Space (PS)
    for i in range(num_layers):
        density = cfg['ps'][i * 2]
        weight = cfg['ps'][i * 2 + 1]
        for key in models[0].state_dict().keys():
            if 'model' in key:
                merged_model.state_dict()[key] = density * models[0].state_dict()[key] + (1 - density) * models[1].state_dict()[key]

    # Merge in the Data Flow Space (DFS)
    for t in range(num_layers * 3):  # adjusted to `num_layers * 3` from max_path_length for clarity
        for m in range(len(models)):
            if cfg['dfs']['include_layers'][t * len(models) + m]:
                layer_index = t % num_layers
                for key in models[m].state_dict().keys():
                    if 'model' in key:
                        merged_model.state_dict()[key] = models[m].state_dict()[key] * cfg['dfs']['W'][layer_index][m]
    
    merged_model.save_pretrained(res)
    logging.info("Model merging completed successfully.")
    return res

class EvaluationStrategyBase(ABC):
    def __init__(self, config, num_gpus=None, batch_size=None, task_search_path=None):
        self.config = config
        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.batch_size = batch_size
        self.task_manager = tasks.TaskManager(include_path=task_search_path)

    @abstractmethod
    def evaluate_genotypes(self, genotypes: List[np.ndarray], models) -> List[float]:
        pass

    @abstractmethod
    def evaluate_genotype(self, genotype: np.ndarray, models) -> float:
        pass

class SerialEvaluationStrategy(EvaluationStrategyBase):
    def __init__(self, config, model_storage_path: Optional[str] = None, num_gpus=None, batch_size=None, task_search_path=None):
        super().__init__(config, num_gpus=num_gpus, batch_size=batch_size, task_search_path=task_search_path)
        self.model_storage_path = model_storage_path

    def evaluate_genotypes(self, genotypes: List[np.ndarray], models) -> List[float]:
        scores = []
        for genotype in genotypes:
            merged_path = merge_model(genotype, models, self.model_storage_path)
            try:
                model_args = {"pretrained": merged_path, "dtype": "bfloat16", "use_cache": True}
                score = evaluate_model(
                    "huggingface",
                    self.config.tasks,
                    model_args,
                    self.task_manager,
                )["score"]
                scores.append(score)
            finally:
                shutil.rmtree(merged_path)
        return scores

    def evaluate_genotype(self, genotype: np.ndarray, models) -> float:
        return self.evaluate_genotypes([genotype], models)[0]

def evolve_ps(models, task_names, population_size, generations, batch_size, limit, evaluation_strategy):
    num_layers = models[0].config.num_hidden_layers
    ps_genotype_length = num_layers * 2
    config_space = get_ps_config_space(num_layers)
    
    optimizer = CMA(mean=np.zeros(ps_genotype_length), sigma=1.0, population_size=population_size)
    best_scores = {task_name: float('-inf') for task_name in task_names}
    best_model = None
    logging.debug(f"Optimizer configured with population size {population_size} and PS genotype length {ps_genotype_length}")
    try:
        for _ in range(generations):
            genotypes = [optimizer.ask() for _ in range(population_size)]
            logging.debug(f"Generated PS genotypes of length {len(genotypes[0]) if genotypes else 'None'}")
            scores = evaluation_strategy.evaluate_genotypes(genotypes, models)
            solutions = list(zip(genotypes, scores))
            optimizer.tell(solutions)
            for genotype, score in solutions:
                if all(score > best_scores[task_name] for task_name in task_names):
                    best_scores = {task_name: score for task_name in task_names}
                    best_model = merge_model(genotype, models, evaluation_strategy.model_storage_path)
        logging.info("Evolutionary search in PS completed successfully.")
    except Exception as e:
        logging.error(f"Failed in evolve_ps: {e}")
        raise
    return best_model, best_scores


def evolve_dfs(models, task_names, population_size, generations, batch_size, limit, evaluation_strategy):
    total_layers = sum(model.config.num_hidden_layers for model in models)
    num_models = len(models)
    max_path_length = total_layers * 3
    dfs_genotype_length = total_layers * max_path_length + total_layers * num_models
    config_space = get_dfs_config_space(total_layers, num_models, max_path_length)
    
    optimizer = CMA(mean=np.zeros(dfs_genotype_length), sigma=1.0, population_size=population_size)
    best_scores = {task_name: float('-inf') for task_name in task_names}
    best_model = None

    logging.debug(f"Optimizer configured for DFS with population size {population_size} and DFS genotype length {dfs_genotype_length}")
    
    try:
        for _ in range(generations):
            genotypes = [optimizer.ask() for _ in range(population_size)]
            logging.debug(f"Generated DFS genotypes of length {len(genotypes[0]) if genotypes else 'None'} for each of {population_size} samples")

            # Check genotype lengths right after generation
            for genotype in genotypes:
                if len(genotype) != dfs_genotype_length:
                    logging.error(f"DFS Genotype generation error: Expected length {dfs_genotype_length}, got {len(genotype)}")
                    raise ValueError(f"DFS Genotype length mismatch: Expected {dfs_genotype_length}, got {len(genotype)}")
            
            scores = evaluation_strategy.evaluate_genotypes(genotypes, models)
            solutions = list(zip(genotypes, scores))
            optimizer.tell(solutions)

            for genotype, score in solutions:
                if all(score > best_scores[task_name] for task_name in task_names):
                    best_scores = {task_name: score for task_name in task_names}
                    best_model = merge_model(genotype, models, evaluation_strategy.model_storage_path)

        logging.info("Evolutionary search in DFS completed successfully.")
    except Exception as e:
        logging.error(f"Failed in evolve_dfs: {e}")
        raise
    
    return best_model, best_scores

def merge_ps_dfs(models, task_names, ps_population_size, ps_generations, dfs_population_size, dfs_generations, batch_size, limit, output_path):
    try:
        evaluation_strategy = SerialEvaluationStrategy(
            config=EvolMergeConfiguration(tasks=task_names),
            model_storage_path=output_path,
            batch_size=batch_size,
        )
        ps_merged_model, ps_scores = evolve_ps(models, task_names, ps_population_size, ps_generations, batch_size, limit, evaluation_strategy)
        if ps_merged_model is None:
            raise ValueError("PS merging failed, cannot proceed with DFS merging.")
        dfs_merged_model, dfs_scores = evolve_dfs(models + [ps_merged_model], task_names, dfs_population_size, dfs_generations, batch_size, limit, evaluation_strategy)
        if dfs_merged_model:
            tokenizer = AutoTokenizer.from_pretrained(models[0].config._name_or_path)
            dfs_merged_model.save_pretrained(output_path)
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
        
        # Directly loading tasks from the YAML file
        with open("test_options.yaml", 'r') as file:
            tasks_config = yaml.safe_load(file)
            tasks_list = tasks_config.get('tests', [])
        
        tasks_vars = {task: tk.BooleanVar(value=False) for task in tasks_list}
        for task in tasks_list:
            tk.Checkbutton(window, text=task, variable=tasks_vars[task]).pack()

        # UI elements for configuration settings
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
        logging.info("Task selection window displayed successfully.")
    except Exception as e:
        logging.error(f"Failed in create_task_selection_window: {e}")
        raise

def run_evaluation(models, selected_tasks, batch_size, limit, load_in_4bit, output_path,
                   ps_population_size, ps_generations, dfs_population_size, dfs_generations):
    try:
        merged_model, scores = merge_ps_dfs(models, selected_tasks, ps_population_size=ps_population_size, ps_generations=ps_generations,
                                            dfs_population_size=dfs_population_size, dfs_generations=dfs_generations,
                                            batch_size=batch_size, limit=limit, output_path=output_path)
        logging.info(f"Best merged model saved to '{output_path}'. Scores: {scores}")

        # Create a pandas DataFrame from the scores dictionary
        df_scores = pd.DataFrame.from_dict(scores, orient='index', columns=['Score'])
        print("Scores DataFrame:")
        print(df_scores)

    except Exception as e:
        logging.error(f"Failed in run_evaluation: {e}")
        raise

def main():
    try:
        model_folders = select_model_folders()
        if len(model_folders) != 3:
            raise ValueError("Incorrect folder selection. Please select exactly two model folders and one output folder.")
        output_folder = model_folders.pop()
        create_task_selection_window(model_folders, output_folder)
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()