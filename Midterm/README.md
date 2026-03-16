# Midterm: Faster R-CNN HPO with Optuna + W&B

This folder contains the executed work for the midterm on stage-wise hyperparameter optimization of `torchvision` Faster R-CNN using COCO MiniTrain, Weights & Biases, and final reduced-budget multi-seed retraining.

## Open the files in this order

1. `midterm/MIDTERM_torchvision-frcnn-hpo.ipynb`  
   Main notebook containing:
   - baseline run
   - Stage 1 optimizer tuning
   - Stage 2 RPN tuning
   - Stage 3 RoI-head tuning
   - Stage 4 post-processing calibration
   - final reduced-budget multi-seed retraining
   - written analysis and limitations

2. `midterm/midterm_torchvision_frcnn_hpo.py`  
   Python export of the notebook for reference.

3. `midterm/WANDB_LINK.txt`  
   Contains the W&B project/report link used for plots and experiment tracking.

## What is available in this folder

- Executed notebook with the main experimental pipeline
- Stage-wise Optuna results recorded in W&B
- Final tuned configuration
- Reduced-budget robustness summary across completed seed runs
- Written discussion of runtime limitations and recovery steps

## Important note for the TA

Due to repeated Colab disconnections and runtime resets near the end of the experiment pipeline, some in-memory variables, helper functions, and Optuna study objects were lost after results had already been produced. To continue the workflow without rerunning completed stages, the best completed results from later stages were manually restored in the notebook from the recorded W&B outputs and notebook logs. These adjustments were only used to recover notebook state after runtime interruptions and did not change the completed recorded metric values.

The final multi-seed retraining was also affected by Colab instability. The original longer retraining budget could not be completed before repeated runtime failures, so the final robustness section was completed using the available finished seed runs and a reduced training budget. This limitation is documented in the notebook.

## Main metrics reported

- COCO-style validation mAP
- AP50
- AP75
- reduced-budget seed robustness summary (`mean ± std`) using the completed available runs

## How to grade

The main file for grading is:

`midterm/MIDTERM_torchvision-frcnn-hpo.ipynb`

The TA can use the W&B link for the plots and supporting visual evidence, but the notebook contains the written report, results, and limitation notes needed for grading.