# DevNexus

`DevNexus` is a place for experimental projects and code under development and review.

#### Contents

- neonneon.py is the current Parameter Space (PS) and Data Flow Space (DFS) evolutionary merge effort under development.
  - It is self-contained and includes GUI elements to simplify rapid iteration and troubleshooting.
  - This method will not supercede the existing PS evolutionary merge that has already been accomplished;
     - The DFS effort and best practices are the main focus; PS rework is an effort to incorporate both solutions in one.  

[Based on the SakanaAI Paper](https://arxiv.org/abs/2403.13187)

### Element and Tensor Alignment - Embedding Exfiltration and Alignment + Restructuring Base Model as Merge Method

In the align_tensors scripts; It is up to the user to provide 'phrase_dictionary.txt' and all 4096 desired entries. This is a temporary exploratory measure (developer jank) as all testing has been done with Mistral7B models. Looking into automated parameter filling to match shapes with desired model class and size, likely carried out by a lightweight LLM - the eventual plan is the user provides 64 custom entries of 64 features they want extracted across a paradigm of models and the small LLM scales up the content to match whatever the current models' demand is for shape size. This is to limit heuristic signal loss. For now, Orthogonal Procrustes (Option 4) is the preferred method; the rest may be removed in the final script build with the exception of CCA - which is most useful perserving geometric and semantic meaning alignment, but time consuming and sensitive to proper input. The included scripts are rapidly iterated on; please open PRs as desired, feedback highly warrented and encouraged.
