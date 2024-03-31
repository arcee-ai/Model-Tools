# Model-Tools

This repository is an assorted toolkit of python scripts and systems designed for model diagnostics, test & validation, as well overall convenience tools that focus on streamlining common operations.

---

## Table of Contents:
- [EasyPEFTGUIPro](#easypeftguipro)
- [More Tools...](#more-tools)

---

### EasyPEFTGUIPro

`EasyPeftGUIPro` is designed to simplify the process of fusing a language model with a compatible LoRA, then saving the resulting model to a user's directory of choice.

#### Prerequisites

- Python 3.6 or newer
- `transformers` library
- `torch` library
- `tkinter` library

#### Usage

1. Clone this repository
2. Navigate to the repository folder
3. Run the `easypeftguipro.py` script
4. Follow the on-screen prompts to select the base model, the desired LoRA, and an output directory to save the result.

#### Considerations

This script relies on the `tkinter` library for interactive menus. An OS with a GUI interface is required.
LoRAs made for a different model architecture or B number of parameters than the base language model selected will not successfully merge.
LoRAs can however be merged with any pretrained model based on the same architecture and B of parameter size.

---

# Model Evaluation Workflow

<p align="center">
Follow the steps below for evaluating your model using the Evaluation GUI.
</p>

<p align="center">
<table>
  <tr>
    <td><center><img src="media/evalgui/1_modelselect.png" width="300"/></center>center></td>
    <td><center><img src="media/evalgui/2_evalselect.png" height="225"/></center></td> <!-- Adjust height to match the adjacent image -->
  </tr>
  <tr>
    <td align="center">Select Model Folder</td>
    <td align="center">Select Eval Sets</td>
  </tr>
  <tr>
    <td><img src="media/evalgui/3_evalprocess.png" width="300"/></td>
    <td><img src="media/evalgui/4_evalnotification.png" width="300"/></td>
  </tr>
  <tr>
    <td align="center">Evaluation Process Continues in CLI</td>
    <td align="center">Notification Eval Results Saved</td>
  </tr>
  <tr>
    <td colspan="2"><img src="media/evalgui/5_evalresults.png" width="600"/></td>
  </tr>
  <tr>
    <td colspan="2" align="center">Example Evaluation Results</td>
  </tr>
</table>
</p>

<p align="center">
In summary, these steps will guide you through the model evaluation process.
</p>

---

## More Tools Coming Soon...
<!-- Future sections for additional tools will go here -->
