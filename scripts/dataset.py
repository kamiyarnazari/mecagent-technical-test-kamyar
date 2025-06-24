import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import GPT2Tokenizer

class CADCodeDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, image_transform, max_length=256):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.transform = image_transform
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # Image preprocessing
        image = self.transform(sample["image"])

        # Tokenization
        code = sample["cadquery"]
        tokenized = self.tokenizer(
            code,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "image": image,
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
        }

# Optional helper
def get_default_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
