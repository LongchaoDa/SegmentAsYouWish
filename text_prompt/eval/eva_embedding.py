import torch
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPTokenizer, CLIPTextModel
import json
from sklearn.metrics import classification_report, silhouette_score
import matplotlib.pyplot as plt
import umap
import numpy as np
from segment_anything import sam_model_registry
from segment_anything.modeling import PromptEncoder
from os.path import join, basename
from tqdm import tqdm
from copy import deepcopy
from time import time
from util import get_pickle


device = "cuda:0"
batch_size=16
medsam_checkpoint = "/home/ec2-user/textSegment/MedSAM/work_dir/MedSAM/medsam_vit_b_full.pth"
num_workers = 4


global_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")   

# Define the test dataset class
class TestDataset(Dataset):
    def __init__(self, data_path, label_dict):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        self.label_dict = label_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        description, label = self.data[idx]
        tokens = self.tokenize_text(description)
        return {
            'description': description,
            'tokens': tokens,
            'label': label,
        }

    def tokenize_text(self, text):
        return self.tokenizer(
            text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids.squeeze(0)

# Define the text prompt encoder class (as in your training script)
class TextPromptEncoder(PromptEncoder):
    def __init__(self, embed_dim=256, image_embedding_size=(64, 64), input_image_size=(1024, 1024), mask_in_chans=1, activation=nn.GELU):
        super().__init__(embed_dim, image_embedding_size, input_image_size, mask_in_chans, activation)
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
        text_encoder.requires_grad_(False)
        self.text_encoder = text_encoder
        self.text_encoder_head = nn.Linear(512, embed_dim)

    def forward(self, points, boxes, masks, tokens):
        bs = self._get_batch_size(points, boxes, masks, tokens)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        if tokens is not None:
            with torch.no_grad():
                encoder_hidden_states = self.text_encoder(tokens)[0]
            text_embeddings = self.text_encoder_head(encoder_hidden_states)
            sparse_embeddings = torch.cat([sparse_embeddings, text_embeddings], dim=1)
        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(bs, -1, self.image_embedding_size[0], self.image_embedding_size[1])
        return sparse_embeddings, dense_embeddings

    def _get_batch_size(self, points, boxes, masks, tokens):
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        elif tokens is not None:
            return tokens.shape[0]
        else:
            return 1

# Define the MedSAM model class (as in your training script)
class MedSAM(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder, num_classes=13, freeze_image_encoder=True):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.text_classification_head = nn.Linear(prompt_encoder.embed_dim, num_classes)

        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.prompt_encoder.text_encoder_head.parameters():
            param.requires_grad = True
        for param in self.text_classification_head.parameters():
            param.requires_grad = True

        self.freeze_image_encoder = freeze_image_encoder
        if self.freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

    def forward(self, image, tokens):
        with torch.no_grad():
            image_embedding = self.image_encoder(image)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(points=None, boxes=None, masks=None, tokens=tokens)
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding, image_pe=self.prompt_encoder.get_dense_pe(), 
            sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings, multimask_output=False
        )
        with torch.no_grad():
            text_embeddings = self.prompt_encoder.text_encoder_head(self.prompt_encoder.text_encoder(tokens)[0])
        text_classification_logits = self.text_classification_head(text_embeddings)
        text_classification_logits, _ = text_classification_logits.max(dim=1)
        return low_res_masks, text_classification_logits, text_embeddings

def aggregate_pkl_data(pkl_data):
    aggregated_data = {}
    list_all = list(pkl_data['s'].keys())
    for key in list_all:
        combined_list = []
        for subset in pkl_data.values():
            combined_list.extend(subset[key])
        aggregated_data[key] = combined_list
    return aggregated_data


class CustomDataset(Dataset):
    def __init__(self, data_dict):
        self.labels = list(data_dict.keys())
        self.data = []
        self.targets = []
        for label, features in data_dict.items():
            for feature in features:
                self.data.append(feature)  # Assume features are descriptions
                self.targets.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = self.data[idx]
        label = self.targets[idx]
        return feature, self.labels.index(label)


# Define paths and load the model (sam)
model_path = "/home/ec2-user/textSegment/Proposal/MedSAM/extensions/text_prompt/train_text_prompt_max_200/medsam_text_prompt_best_max_200.pth"

# This is the test_data i constructed: (all text)
test_data_path = "/home/ec2-user/textSegment/Proposal/MedSAM/extensions/text_prompt/eval/full_combine.pkl"
pkl_data = get_pickle(test_data_path)
test_data = aggregate_pkl_data(pkl_data)
dataset = CustomDataset(test_data)# total: 13*120 = 1560; (120: liver, 120: right kidney)... 

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Initialize the model
sam_model = sam_model_registry["vit_b"](checkpoint=medsam_checkpoint)
text_prompt_encoder = TextPromptEncoder(embed_dim=256, image_embedding_size=(64, 64), input_image_size=(1024, 1024), mask_in_chans=1, activation=nn.GELU)
medsam_prompt_encoder_state_dict = sam_model.prompt_encoder.state_dict()
for keys in text_prompt_encoder.state_dict().keys():
    if keys in medsam_prompt_encoder_state_dict.keys():
        text_prompt_encoder.state_dict()[keys] = deepcopy(medsam_prompt_encoder_state_dict[keys])
medsam_model = MedSAM(image_encoder=sam_model.image_encoder, mask_decoder=deepcopy(sam_model.mask_decoder), prompt_encoder=text_prompt_encoder, freeze_image_encoder=True)

# Load the trained weights
medsam_model.load_state_dict(torch.load(model_path)["model"])
medsam_model.eval()
medsam_model.to(device)

# Load the test dataset
# test_dataset = TestDataset(test_data_path, label_dict)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Collect embeddings and labels
all_labels = []
all_preds = []
all_embeddings = []


with torch.no_grad():
    for batch in dataloader:
        description, label = batch
        description = description[0]  # Get the actual text description

        # Assuming medsam_model has a method to convert description to tokens
        tokens = global_tokenizer.tokenize(description).to(device)

        # Run the model to get embeddings
        _, _, text_embeddings = medsam_model(None, tokens)

        # Store the embeddings and labels
        all_embeddings.append(text_embeddings.cpu().numpy())
        all_labels.append(label.item())

# Convert embeddings to numpy array
all_embeddings = np.array(all_embeddings)

# Apply UMAP
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
umap_embeddings = umap_model.fit_transform(all_embeddings)

# Calculate Silhouette Score
sil_score = silhouette_score(umap_embeddings, all_labels)
print(f'Silhouette Score: {sil_score}')

# Plot UMAP
plt.figure(figsize=(10, 8))
for label in np.unique(all_labels):
    indices = np.where(np.array(all_labels) == label)
    plt.scatter(umap_embeddings[indices, 0], umap_embeddings[indices, 1], label=f'Class {label}', alpha=0.7)
plt.title(f'UMAP projection (Silhouette Score: {sil_score:.2f})')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.legend()
plt.show()

# Print the classification report
print(classification_report(all_labels, all_preds, target_names=[str(label) for label in label_dict.keys()]))
