import torch
import torch.nn as nn
import numpy as np
from .segment_anything.modeling import PromptEncoder
from transformers import CLIPTextModel

# Text Prompt Encoder class
class TextPromptEncoder(PromptEncoder):
    def __init__(
        self,
        embed_dim = 256,
        image_embedding_size = (64, 64),
        input_image_size = (1024, 1024),
        mask_in_chans = 1,
        activation = nn.GELU,
        ) -> None:
        super().__init__(embed_dim, image_embedding_size, input_image_size, mask_in_chans, activation)
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
        text_encoder.requires_grad_(False)
        self.text_encoder = text_encoder
        self.text_encoder_head = nn.Linear(512, embed_dim)

    def forward(
        self, points,
        boxes,
        masks,
        tokens,
    ):
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          tokens (torch.Tensor or none): text tokens to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks, tokens)
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device()
        )
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        if tokens is not None:
            with torch.no_grad():
                encoder_hidden_states = self.text_encoder(tokens)[0]
            text_embeddings = self.text_encoder_head(encoder_hidden_states)
            sparse_embeddings = torch.cat([sparse_embeddings, text_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings
    
    def _get_batch_size(self, points, boxes, masks, tokens):
        """
        Returns the batch size of the inputs.
        """
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
        
        

# MedSAM With Canonicalization model class
class MedSAMWithCanonicalization(nn.Module):
    def __init__(self, 
                 image_encoder, 
                 mask_decoder,
                 prompt_encoder,
                 canonicalization_network = None,
                 use_canonicalization = True,
                 use_classify_head = True,
                 text_pooling = "mean",
                 num_classes=13,
                 freeze_image_encoder=True): # default is freezing the image
        super().__init__()
        # add can network
        
        self.canonicalization_network = canonicalization_network
        self.use_canonicalization = use_canonicalization
        self.use_classify_head = use_classify_head
        self.text_pooling = text_pooling

        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # dlc v2: add text classification:
        self.text_classification_head = nn.Linear(prompt_encoder.embed_dim, num_classes)

        # freeze prompt encoder except for text_encoder_head
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.prompt_encoder.text_encoder_head.parameters():
            param.requires_grad = True
        # dlc v2: add text classification:
        for param in self.text_classification_head.parameters():
            param.requires_grad = True
        
        self.freeze_image_encoder = freeze_image_encoder
        if self.freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

    def forward(self, image, tokens):
        # do not compute gradients for image encoder
        if self.use_canonicalization:
            # print("used canonicalization")
            canonicalized_image = self.canonicalization_network(image)
        # with torch.no_grad():
            image_embedding = self.image_encoder(canonicalized_image) # (B, 256, 64, 64)
        else: 
            # with torch.no_grad():
            image_embedding = self.image_encoder(image) # (B, 256, 64, 64)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
            tokens=tokens,
        )

        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          ) # (B, 1, 256, 256)
        
        # text_embeddings = sparse_embeddings[:, -1, :]  # 提取文本嵌入
        if self.use_classify_head:
            with torch.no_grad():
                text_embeddings = self.prompt_encoder.text_encoder_head(
                    self.prompt_encoder.text_encoder(tokens)[0]
                )  # 提取文本嵌入
            text_classification_logits = self.text_classification_head(text_embeddings)
            if self.text_pooling == "mean":
                # 对 logits 进行mean poolinhg
                text_classification_logits = text_classification_logits.mean(dim=1)
            elif self.text_pooling == "max":
                # max pooling
                text_classification_logits, _ = text_classification_logits.max(dim=1)
            else: # by default: 
                text_classification_logits, _ = text_classification_logits.mean(dim=1)
        else: 
        # not using the classification head
            text_classification_logits = None
        if self.use_canonicalization:
            low_res_masks = self.canonicalization_network.invert_canonicalization(low_res_masks)
        
        return low_res_masks, text_classification_logits
