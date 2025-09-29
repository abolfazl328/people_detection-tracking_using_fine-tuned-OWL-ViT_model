import os
import json
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from PIL import Image

from transformers import (
    OwlViTProcessor,
    OwlViTForObjectDetection,
    TrainingArguments,
    Trainer,
    get_scheduler,
)

# ===============================
# CONFIG - ANTI-OVERFITTING MODIFICATIONS
# ===============================
DATASET_PATH = "./datasets/human_detection_tracking"
ANNOTATION_FILE = "annotations_coco.json"
MODEL_CHECKPOINT = "google/owlvit-base-patch16"
OUTPUT_DIR = "./owlvit_finetuned_person_v3"

# 1. Reduced epochs and lower learning rate
NUM_EPOCHS = 15
LR = 5e-7
WARMUP_RATIO = 0.1

# 2. Use only ONE text query but with different variations across epochs
# We'll handle this in the dataset class
BASE_TEXT_QUERY = "a person"

# ===============================
# LOAD COCO JSON
# ===============================
with open(os.path.join(DATASET_PATH, ANNOTATION_FILE), "r") as f:
    coco = json.load(f)

images_info = coco["images"]
annotations = coco["annotations"]

ann_by_img_id = {}
for ann in annotations:
    ann_by_img_id.setdefault(ann["image_id"], []).append(ann)

train_info, val_info = train_test_split(images_info, test_size=0.2, random_state=42)

# ===============================
# DATASET CLASS WITH TEXT VARIATIONS
# ===============================
class CocoPersonDataset(Dataset):
    def __init__(self, image_info_list, annotations_dict, image_base_path, processor):
        self.image_info_list = image_info_list
        self.annotations_dict = annotations_dict
        self.image_base_path = image_base_path
        self.processor = processor
        # Different text queries to prevent overfitting
        self.text_variations = [
            "a person", 
            "human", 
            "person walking", 
            "a human", 
            "person"
        ]
        
    def __len__(self):
        return len(self.image_info_list)

    def __getitem__(self, idx):
        image_info = self.image_info_list[idx]
        
        # Fix the file path
        file_name = image_info["file_name"]
        if file_name.startswith("images/"):
            file_name = file_name.replace("images/", "", 1)
        
        image_path = os.path.join(self.image_base_path, file_name)
        
        if not os.path.exists(image_path):
            alt_path = os.path.join(self.image_base_path, "images", image_info["file_name"])
            if os.path.exists(alt_path):
                image_path = alt_path
            else:
                raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert("RGB")

        anns = copy.deepcopy(self.annotations_dict.get(image_info["id"], []))

        # Prepare bounding boxes and labels
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            # Convert to normalized coordinates [0, 1]
            image_width, image_height = image.size
            boxes.append([x/image_width, y/image_height, (x + w)/image_width, (y + h)/image_height])
            labels.append(0)  # person class

        # Use different text queries to prevent overfitting
        # Use modulo to cycle through variations based on index
        text_query = [self.text_variations[idx % len(self.text_variations)]]

        # Process image and text
        encoding = self.processor(
            images=image,
            text=text_query,  # Single text query per sample
            return_tensors="pt"
        )
        
        # Remove batch dimension
        for k in encoding:
            encoding[k] = encoding[k].squeeze(0)

        # Add target boxes and labels (normalized coordinates)
        encoding["target_boxes"] = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        encoding["target_labels"] = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64)

        return encoding

# ===============================
# PROCESSOR + DATASETS
# ===============================
processor = OwlViTProcessor.from_pretrained(MODEL_CHECKPOINT)

# Remove TEXT_QUERIES from dataset initialization
train_dataset = CocoPersonDataset(train_info, ann_by_img_id, os.path.join(DATASET_PATH, "images"), processor)
val_dataset = CocoPersonDataset(val_info, ann_by_img_id, os.path.join(DATASET_PATH, "images"), processor)

def data_collator(batch):
    batch_out = {}
    
    # Stack image inputs
    batch_out["pixel_values"] = torch.stack([item["pixel_values"] for item in batch])
    batch_out["input_ids"] = torch.stack([item["input_ids"] for item in batch])
    batch_out["attention_mask"] = torch.stack([item["attention_mask"] for item in batch])
    
    # Collect targets
    batch_out["target_boxes"] = [item["target_boxes"] for item in batch]
    batch_out["target_labels"] = [item["target_labels"] for item in batch]
    
    return batch_out

# ===============================
# MODEL
# ===============================
model = OwlViTForObjectDetection.from_pretrained(MODEL_CHECKPOINT)

# Only freeze early layers, allow later layers to fine-tune
for name, param in model.named_parameters():
    if "vision_model.encoder.layers" in name:
        layer_num = int(name.split(".layers.")[1].split(".")[0])
        if layer_num < 6:  # Freeze first 6 layers of vision encoder
            param.requires_grad = False
    elif "text_model.encoder.layers" in name:
        layer_num = int(name.split(".layers.")[1].split(".")[0])
        if layer_num < 6:  # Freeze first 6 layers of text encoder
            param.requires_grad = False

# ===============================
# OPTIMIZER
# ===============================
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1e-8
)

# ===============================
# TRAINING ARGS
# ===============================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    weight_decay=0.01,
    eval_strategy="no",
    save_strategy="epoch",
    logging_steps=10,
    save_total_limit=2,
    remove_unused_columns=False,
    load_best_model_at_end=False,
    fp16=torch.cuda.is_available(),
    dataloader_pin_memory=False,
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,
    warmup_ratio=WARMUP_RATIO,
)

# ===============================
# CUSTOM TRAINER WITH L2 REGULARIZATION
# ===============================
class OWLTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.box_loss_fn = nn.SmoothL1Loss(beta=0.1)
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extract targets
        target_boxes = inputs.pop("target_boxes")
        target_labels = inputs.pop("target_labels")
        
        # Forward pass
        outputs = model(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        
        # Compute detection loss with L2 regularization
        loss = self.compute_detection_loss(outputs, target_boxes, target_labels)
        
        return (loss, outputs) if return_outputs else loss
    
    def compute_detection_loss(self, outputs, target_boxes, target_labels):
        total_loss = 0.0
        batch_size = len(target_boxes)
        
        for i in range(batch_size):
            pred_boxes = outputs.pred_boxes[i]  # (num_queries, 4)
            pred_logits = outputs.logits[i]     # (num_queries, num_classes)
            
            tgt_boxes = target_boxes[i]
            tgt_labels = target_labels[i]
            
            if len(tgt_boxes) == 0:
                # If no targets, use background loss
                bg_loss = self.compute_background_loss(pred_logits)
                total_loss += bg_loss
                continue
            
            # For each target, find best matching prediction
            box_losses = []
            for tgt_box in tgt_boxes:
                # Compute MSE between target box and all predictions
                box_diffs = torch.nn.functional.mse_loss(
                    pred_boxes, 
                    tgt_box.unsqueeze(0).expand_as(pred_boxes), 
                    reduction='none'
                ).mean(dim=1)
                
                # Use the minimum loss (best matching prediction)
                min_box_loss = torch.min(box_diffs)
                box_losses.append(min_box_loss)
            
            # Classification loss - encourage high scores for person class
            person_scores = torch.sigmoid(pred_logits[:, 0])  # Person class scores
            cls_loss = torch.nn.functional.binary_cross_entropy(
                person_scores,
                torch.ones_like(person_scores)  # Target: all should detect person
            )
            
            # Combine losses
            if box_losses:
                avg_box_loss = sum(box_losses) / len(box_losses)
                total_loss += avg_box_loss + cls_loss * 0.1  # Weight classification lower
            else:
                total_loss += cls_loss
        
        # Average over batch
        if batch_size > 0:
            total_loss = total_loss / batch_size
        
        # ADD L2 REGULARIZATION TO PREVENT OVERFITTING
        l2_lambda = 0.01  # Regularization strength
        l2_reg = torch.tensor(0.).to(total_loss.device)
        for param in self.model.parameters():
            if param.requires_grad:
                l2_reg += torch.norm(param)  # L2 norm of weights
        
        total_loss = total_loss + l2_lambda * l2_reg
        
        return total_loss
    
    def compute_background_loss(self, pred_logits):
        """Loss for images with no objects - encourage low confidence"""
        person_scores = torch.sigmoid(pred_logits[:, 0])
        bg_loss = torch.nn.functional.binary_cross_entropy(
            person_scores,
            torch.zeros_like(person_scores)  # Target: no persons
        )
        return bg_loss * 0.1  # Lower weight for background

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = inputs.copy()
        inputs.pop("target_boxes", None)
        inputs.pop("target_labels", None)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

# ===============================
# SCHEDULER
# ===============================
num_training_steps = len(train_dataset) // (2 * training_args.gradient_accumulation_steps) * NUM_EPOCHS
lr_scheduler = get_scheduler(
    name="cosine",
    optimizer=optimizer,
    num_warmup_steps=int(WARMUP_RATIO * num_training_steps),
    num_training_steps=num_training_steps,
)

# ===============================
# TRAINER
# ===============================
trainer = OWLTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,
    data_collator=data_collator,
    processing_class=processor,
    optimizers=(optimizer, lr_scheduler),
)

# ===============================
# TRAIN
# ===============================
print("Starting ANTI-OVERFITTING fine-tuning...")
print("Applied modifications:")
print("1. Multiple text variations across samples: ['a person', 'human', 'person walking', 'a human', 'person']")
print("2. L2 regularization (lambda=0.01)")
print("3. Reduced epochs: 15 (from 20)")
print("4. Lower learning rate: 5e-7 (from 1e-6)")
print("=" * 50)
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training samples: {len(train_dataset)}")
print("=" * 50)

trainer.train()
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print("✅ Anti-overfitting fine-tuning complete! Model saved to", OUTPUT_DIR)