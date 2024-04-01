import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os
import platform
from datetime import datetime

# Environment variables setup
os.environ['CUDA_LAUNCH_BLOCKING'] = "True"
os.environ['HF_DATASETS_ALLOW_CODE_EXECUTION'] = "True"
os.environ['ALLOW_REMOTE_CODE'] = "True"

# Ensure that the selected model folder is accessible across functions
model_folder = ""

def select_model_folder():
    global model_folder
    root = tk.Tk()
    root.withdraw()
    model_folder = filedialog.askdirectory(title="Select Model to Evaluate")
    root.destroy()  # Ensure the root window is destroyed after folder selection
    return model_folder

def create_task_selection_window():
    global model_folder
    tasks_list = [
        "ai2_arc", "hellaswag", "mmlu", "winogrande", "gsm8k",
    ]

    window = tk.Tk()
    window.title("Select Tasks")
    window.withdraw()  # Withdraw the window as it's immediately created

    tasks_vars = {task: tk.BooleanVar() for task in tasks_list}

    for task in tasks_list:
        tk.Checkbutton(window, text=task, variable=tasks_vars[task]).pack()

    def on_evaluate_click():
        selected_tasks = [task for task, var in tasks_vars.items() if var.get()]
        if not selected_tasks:
            messagebox.showerror("Error", "No tasks selected. Please select at least one task.")
            return
        window.destroy()
        run_evaluation(selected_tasks)

    tk.Button(window, text="Evaluate", command=on_evaluate_click).pack()

    window.deiconify()  # Make the window visible now after all widgets are added
    window.mainloop()

def run_evaluation(selected_tasks):
    global model_folder
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
        output = subprocess.check_output(command, universal_newlines=True)
        results_start = output.find("|     Tasks      |Version|Filter|n-shot| Metric |Value |   |Stderr|")
        results_end = output.find("\n\n", results_start)
        results = output[results_start:results_end].strip()

        model_name = os.path.basename(model_folder)
        current_date = datetime.now().strftime("%Y-%m-%d")
        file_name = f"{model_name}_{current_date}_{'_'.join(selected_tasks)}.txt"

        with open(file_name, "w") as file:
            file.write(results)

        show_custom_dialog(file_name)

    except subprocess.CalledProcessError as e:
        messagebox.showerror("Evaluation Error", f"An error occurred during evaluation: {e}")

def show_custom_dialog(file_name):
    dialog = tk.Tk()
    dialog.title("Model Evaluation Complete")
    dialog.geometry('464x175')

    tk.Label(dialog, text=f"Results have been saved to {file_name}",
             justify=tk.CENTER, font=('TkDefaultFont', 11)).pack(expand=True)

    def open_results():
        try:
            if platform.system() == 'Darwin':
                subprocess.run(['open', file_name])
            elif platform.system() == 'Windows':
                os.startfile(file_name)
            else:
                subprocess.run(['xdg-open', file_name])
        except Exception as e:
            messagebox.showerror("Open Results", f"Failed to open the file: {e}")
        finally:
            dialog.destroy()

    button_frame = tk.Frame(dialog)
    button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=20)

    tk.Button(button_frame, text="Open Results", width=25, command=open_results).grid(row=0, column=0, padx=10)
    tk.Button(button_frame, text="Close/Acknowledge", width=25, command=dialog.destroy).grid(row=0, column=1, padx=10)

    dialog.mainloop()

def main():
    if select_model_folder():
        create_task_selection_window()

if __name__ == "__main__":
    main()
