import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights

class BaselineGenCAD(nn.Module):
    def __init__(self, decoder_model, embedding_dim=768):
        super().__init__()

        # 1. Vision Encoder: Pretrained ResNet-18 (outputs 512-dim)
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)        
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # adapt for grayscale
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-1])  # remove final FC

        # 2. Project image embedding to transformer input dim
        self.projection = nn.Linear(512, embedding_dim)

        # 3. Decoder: GPT-like model
        self.decoder = decoder_model  # AutoModelForCausalLM (e.g. GPT2)

    def forward(self, images, decoder_input_ids, attention_mask=None):
        # Extract image features
        features = self.vision_encoder(images)  # [B, 512, 1, 1]
        features = features.view(features.size(0), -1)  # [B, 512]
        visual_embedding = self.projection(features)    # [B, 768]

        # Add visual embedding as prefix to input embeddings
        inputs_embeds = self.decoder.transformer.wte(decoder_input_ids)  # [B, seq_len, 768]
        visual_embedding = visual_embedding.unsqueeze(1)                 # [B, 1, 768]
        decoder_inputs = torch.cat([visual_embedding, inputs_embeds], dim=1)  # prepend image token

        # Adjust attention mask if provided
        if attention_mask is not None:
            visual_mask = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype).to(attention_mask.device)
            attention_mask = torch.cat([visual_mask, attention_mask], dim=1)

        labels = decoder_input_ids.clone()
        output = self.decoder(
        inputs_embeds=decoder_inputs,
        attention_mask=attention_mask,
        labels=torch.cat([torch.full((labels.size(0), 1), -100).to(labels.device), labels], dim=1)
        )


        return output
