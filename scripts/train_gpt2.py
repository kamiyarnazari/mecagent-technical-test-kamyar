import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import BaselineGenCAD
from dataset import CADCodeDataset, get_default_transform
from datasets import load_dataset
import wandb

from metrics.valid_syntax_rate import evaluate_syntax_rate
from metrics.best_iou import evaluate_codes


config = {
    "model_name": "gpt2",
    "batch_size": 8,
    "epochs": 10,
    "lr": 5e-5,
    "max_length": 256,
    "project": "mecagent-gencad",
    "patience": 2,
    "min_iou_delta": 0.01,
    "eval_samples": 50,
    "eval_every": 50
}

# Init Weights & Biases
wandb.init(project=config["project"], config=config)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

# Load Dataset
train_ds_raw, test_ds_raw = load_dataset("CADCODER/GenCAD-Code", split=["train", "test"])

# Image transform
image_transform = get_default_transform()

# Build Dataset and Loader
train_ds = CADCodeDataset(train_ds_raw, tokenizer, image_transform)
train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)

# Load Model
decoder_model = AutoModelForCausalLM.from_pretrained(config["model_name"])
model = BaselineGenCAD(decoder_model).cuda()


optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

def evaluate_model(model, test_ds_raw, tokenizer, transform, max_samples=50):
    print("Running evaluation...")
    model.eval()
    gt_dict = {}
    pred_dict = {}

    with torch.no_grad():
        for i in range(min(max_samples, len(test_ds_raw))):
            sample = test_ds_raw[i]
            gt_dict[str(i)] = sample["cadquery"]
            image = transform(sample["image"]).unsqueeze(0).cuda()

            prompt = "Generate the CADQuery code needed to create the CAD for the provided image. Just the code, no other words."
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(image.device)

            for _ in range(config["max_length"]):
                outputs = model(image, input_ids)
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                if next_token.item() == tokenizer.eos_token_id:
                    break

            code = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            code = code.replace(prompt, "").strip()

            
            if "loop1" in code and "loop1 =" not in code:
                code = code.replace("loop1", "loop0")

            pred_dict[str(i)] = code

            if i < 3:
                print(f" Sample {i} Code:\n{code}\n---")

    vsr = evaluate_syntax_rate(pred_dict, verbose=False)
    iou = evaluate_codes(gt_dict, pred_dict)

    print(f" Evaluation done: VSR={vsr['vsr']:.3f}, IoU_best={iou['iou_best']:.3f}")
    return vsr["vsr"], iou["iou_best"]

# Training Loop
best_iou = -1.0
wait = 0
step = 0
early_stop = False

model.train()
for epoch in range(config["epochs"]):
    if early_stop:
        break
    total_loss = 0
    for batch in train_loader:
        step += 1

        images = batch["image"].cuda()
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()

        outputs = model(images, input_ids, attention_mask)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        wandb.log({"loss": loss.item(), "step": step})

        if step > 200 and step % config["eval_every"] == 0:
            print(f"\n Evaluating at step {step}...")
            vsr_score, iou_score = evaluate_model(
                model, test_ds_raw, tokenizer, image_transform, max_samples=config["eval_samples"])

            wandb.log({
                "vsr": vsr_score,
                "iou_best": iou_score,
                "step": step
            }, commit=True)

            if iou_score > best_iou + config["min_iou_delta"]:
                best_iou = iou_score
                wait = 0
                torch.save(model.state_dict(), "gencad_best.pt")
                print(" New best model saved.")
            else:
                wait += 1
                print(f" No improvement. Patience: {wait}/{config['patience']}")
                if wait >= config["patience"]:
                    print(" Early stopping triggered.")
                    early_stop = True
                    break

    avg_loss = total_loss / len(train_loader)
    print(f"\n Epoch {epoch+1}/{config['epochs']} | Avg Loss: {avg_loss:.4f}")
    wandb.log({"avg_loss": avg_loss, "epoch": epoch + 1})

print("\n Training complete. Best IoU_best:", best_iou)
