# Model-Tools
This repository is an assorted toolkit of python scripts and systems designed for model diagnostics, test & validation, as well overall convenience tools that focus on streamlining common operations.

 <br> 

# EasyPEFTGUIPro

EasyPeftGUIPro is designed to simplify the process of fusing a language model with a compatible LoRA, then saving the resulting model to a user's directory of choice.

## Prerequisites

- Python 3.6 or newer
- `transformers` library
- `torch` library
- `tkinter` library

## Usage

1. Clone this repository
2. Navigate to the repository folder
3. Run the `easypeftguipro.py` script
4. Follow the on-screen prompts to select the base model, the desired LoRA, and an output directory to save the result.

## Considerations

This script relies on the tkinter library for interactive menus. An OS with a GUI interface is required.
LoRAs made for a different model architecture or B number of parameters than the base language model selected will not successfully merge.
LoRAs can however be merged with any pretrained model based on the same architecture and B of parameter size.

 <br> 

 
