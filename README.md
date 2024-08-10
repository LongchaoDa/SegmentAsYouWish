# SegmentAsYouWish


## Requirements
Uses the CLIP model from [Huggingface transformers](https://huggingface.co/docs/transformers/index). To install Huggingface transformers:
```
pip install transformers
```

## Training

1. for multi-GPU training:
```
python train_text_prompt_distribute_textloss_processlogit_mean.py \
    -i /home/ec2-user/textSegment/MedSAM/data/npy/CT_Abd \
    -medsam_checkpoint /home/ec2-user/textSegment/MedSAM/work_dir/MedSAM/medsam_vit_b_full.pth \
    -work_dir ./test \
    -num_workers 4 \
    -batch_size 4
```


explanations: 

- `organ_labels_pure.json` is used for training only on labels 
- `organ_labels.json` is used for training on augmented text descriptions

## Testing

1. Use `textPromptSeg.ipynb`: please replaced the path of `medsam_text_demo_checkpoint` with your saved model's weight
2. Please change the `demo_file_nii` with the file to be tested on the text-based segmentation.


## Evaluation

1. Please go to folder at path: `/home/ec2-user/textSegment/Proposal/MedSAM/text_prompt/eval` for evaluation scripts

"/home/ec2-user/textSegment/Proposal/MedSAM/text_prompt/eval/eva_embedding copy_acc.py" is a stable version to evaluate models trained from `train_text_prompt_distribute_textloss_processlogit_mean` and `train_text_prompt_distribute_textloss_processlogit_max.py`, for other script like `train_text_prompt.py` and `train_text_prompt_distribute.py` might need adjust at the model's loading part since they don't have the text-classification layer. 
