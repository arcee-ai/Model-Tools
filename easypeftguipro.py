from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import tkinter as tk
from tkinter import filedialog
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

def select_path(title):
    """
    Opens a dialog for directory selection and returns the selected path.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    path = filedialog.askdirectory(title=title)
    if not path:
        logging.error("No directory selected for '%s'", title)
        raise ValueError(f"No directory selected for '{title}'")
    return path

def get_args():
    """
    Prompts the user to select directories for base model, PEFT adapter, and output.
    """
    try:
        base_model_name_or_path = select_path("Select the base model")
        peft_model_path = select_path("Select PEFT adapter")
        output_dir = select_path("Select output dir")
        return base_model_name_or_path, peft_model_path, output_dir
    except ValueError as e:
        logging.exception("Failed to get directories: %s", e)
        exit(1)

def load_and_merge_models(base_model_path, peft_model_path, output_dir):
    """
    Loads the base model and PEFT adapter, merges them, and saves the result to the specified directory.
    """
    try:
        logging.info("Loading base model from %s", base_model_path)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float32)

        logging.info("Loading PEFT model from %s", peft_model_path)
        model_with_peft = PeftModel.from_pretrained(base_model, peft_model_path)
        model_with_peft = model_with_peft.merge_and_unload()

        logging.info("Saving merged model to %s", output_dir)
        model_with_peft.save_pretrained(output_dir)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        tokenizer.save_pretrained(output_dir)
    except Exception as e:
        logging.exception("An error occurred during model processing: %s", e)
        exit(1)

def main():
    base_model_path, peft_model_path, output_dir = get_args()
    load_and_merge_models(base_model_path, peft_model_path, output_dir)
    logging.info("Process completed successfully. Model saved to %s", output_dir)

if __name__ == "__main__":
    main()
