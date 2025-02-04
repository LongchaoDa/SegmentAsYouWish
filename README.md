# Segment Medical Images with Free Form Language Prompts

## **Abstract**
Medical imaging is crucial for diagnosing a patient’s health condition, and accurate segmentation of these images is essential for isolating regions of interest to ensure precise diagnosis and treatment planning. Existing methods primarily rely on bounding boxes or point-based prompts, while few have explored text-related prompts, despite clinicians often describing their observations and instructions in natural language. To address this gap, we first propose a RAG-based free-form text prompt generator, that leverages the domain corpus to generate diverse and realistic descriptions. Then, we introduce FreeSeg, a novel medical image segmentation model that handles various free-form text prompts, including anatomy-informed queries and anatomy-agnostic queries. Additionally, our model also incorporates a symmetry-aware canonicalization module to ensure consistent, accurate segmentations across varying scan orientations and reduce confusion between the anatomical position of an organ and its appearance in the scan. FreeSeg is trained on a large-scale dataset of over 100k medical images and comprehensive experiments demonstrate the model’s superior language understanding and segmentation precision, outperforming SOTA baselines on both in-domain and out-of-domain datasets.

---
* The model is taken as `FreeSeg` or also called FLanS in short for Free-form language segmentation, we will use interchangeably.


## **Examples for Segmentations Deployment**
Please find more recordings on the use cases in folder `./assets/`.

## **Repository Structure**
A breakdown of the key files and directories in this repository:

- **`configs/`**: Configuration files for training and evaluation.
  - `train_config_main1.yaml`: Main configuration file for our model training.
  - `train_data_paths.json`: Paths to all training datasets.
  - `data_paths_can.json`: Paths to all datasets for training the canonicalizer.
  - `train_pos_data_paths.json`: Paths to datasets with anatomy-agnostic prompts.

- **`data/`**: Datasets and scripts for data preprocessing.
  - `extract_position_npy.py`: Script to generate positional prompts based on organ location.
  - `touse/`: Dataset folder structure.

- **`modules/`**: Model definitions and architecture.
  - `canonicalization_sam/`: Canonicalization module and equivariant networks.
  - `model.py`: Main model module.
  - `segment_anything/`: SAM modules.

- **`utils/`**: Utility scripts for various tasks.
  - `train_utils.py`: PyTorch datasets and training helper functions.
  - `eval_utils.py`: Evaluation functions.
  - `group_utils.py`: Group transformation functions.

- **`prompts/`**: Free-form text prompts.
  - `free_form_text_prompts.json`: Anatomy-informed text prompts.
  - `positional_free_form_text_prompts.json`: Anatomy-agnostic text prompts.
  - `organ_positions.json`: Positional descriptions for organs in datasets equipped with anatomy-agnostic test sets.
  - `organ_bounding_boxes.json`: Ground truth bounding boxes for datasets equipped with anatomy-agnostic test sets.

- **`notebooks/`**: Jupyter notebooks for visualizing predictions.

- **`main_train_three_stage_ddp.py`**: Main script for distributed training using a three-stage process.
- **`main_train_canonicalizer.py`**: Main script for training the canonicalizer.
- **`requirements.txt`**: List of Python dependencies required for this project.

---

## **Installation**

```bash
pip install -e .
```

---
    
## **Training**
**Step 1: Warm up the Canonicalizer**

To warm up the canonicalizer, use the following command:
```bash
python main_train_canonicalizer.py --config configs/train_config_main1.yaml
```
Alternatively, use distributed data parallel (DDP) with:
```bash
sh run_canonicalizer.sh
```

**Step 2 and 3: Train the Canonicalizer**

To train the our model together with Canonicalizer, run the following command:
```bash
python main_train_three_stage_ddp.py --config configs/train_config_main.yaml
```
or
```bash
sh run_main.sh
```

## **Evaluation**
To evaluate our model on all three test sets (FLARE22, WORD, and RAOS), with both anatomy-informed and anatomy-agnostic prompts, run:
```bash
python evals/eval_flans.py
```

`evals` folder also contains evaluation scripts for all the baseline models

## **Checkpoint**
Download pretrained model's weights [here](https://drive.google.com/file/d/1mU_QJYkGYOtXCPDX6iCSQjQvOucmZJF9/view?usp=sharing).



