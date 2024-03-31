import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os
from datetime import datetime

os.environ['CUDA_LAUNCH_BLOCKING'] = "True"
os.environ['HF_DATASETS_ALLOW_CODE_EXECUTION'] = "True"
os.environ['ALLOW_REMOTE_CODE'] = "True"

# Ensure that the selected model folder is accessible across functions
model_folder = ""

def select_model_folder():
    root = tk.Tk()
    root.withdraw()
    global model_folder
    model_folder = filedialog.askdirectory(title="Select Model to Evaluate")
    root.destroy()  # Ensure the root window is destroyed after folder selection
    return model_folder

def create_task_selection_window():
    tasks_list = [
        "ai2_arc", "hellaswag", "mmlu", "winogrande", "gsm8k",
    ]

    window = tk.Tk()
    window.title("Select Tasks")

    # Dictionary to store the BooleanVar for each task
    tasks_vars = {task: tk.BooleanVar() for task in tasks_list}

    for task in tasks_list:
        tk.Checkbutton(window, text=task, variable=tasks_vars[task]).pack()

    def on_evaluate_click():
        # Get the list of selected tasks
        selected_tasks = [task for task, var in tasks_vars.items() if var.get()]
        if not selected_tasks:
            messagebox.showerror("Error", "No tasks selected. Please select at least one task.")
            return
        window.destroy()
        run_evaluation(selected_tasks)

    evaluate_button = tk.Button(window, text="Evaluate", command=on_evaluate_click)
    evaluate_button.pack()

    window.mainloop()

def run_evaluation(selected_tasks):
    print(os.environ.get("HF_DATASETS_ALLOW_CODE_EXECUTION"))

    # Build the command with the selected tasks and model folder
    command = [
        "lm_eval",
        "--model", "hf",
        "--tasks", ",".join(selected_tasks),
        "--batch_size", "24",
        "--limit", "300",
        "--device", "cuda:0",
        "--model_args", f"pretrained={model_folder},load_in_4bit=True"
    ]

    try:
        # Run the evaluation and capture the output
        output = subprocess.check_output(command, universal_newlines=True)

        # Extract the evaluation results from the output
        results_start = output.find("|     Tasks      |Version|Filter|n-shot| Metric |Value |   |Stderr|")
        results_end = output.find("\n\n", results_start)
        results = output[results_start:results_end].strip()

        # Generate the file name based on the model, date, and tests
        model_name = os.path.basename(model_folder)
        current_date = datetime.now().strftime("%Y-%m-%d")
        file_name = f"{model_name}_{current_date}_{'_'.join(selected_tasks)}.txt"

        # Save the evaluation results to a text file
        with open(file_name, "w") as file:
            file.write(results)

        messagebox.showinfo("Evaluation Complete", f"Evaluation results saved to {file_name}")
    except subprocess.CalledProcessError as e:
        error_message = f"An error occurred during evaluation: {e}"
        messagebox.showerror("Evaluation Error", error_message)

def main():
    if select_model_folder():
        create_task_selection_window()
    else:
        print("No model folder selected. Exiting.")

if __name__ == "__main__":
    main()