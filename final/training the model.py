import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, tokenizer, max_length):
        self.image_paths = image_paths
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        text = "placeholder"  # Placeholder text, not used for image processing
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return input_ids, attention_mask, label

def fine_tune_bert_model(train_image_paths, train_labels, val_image_paths, val_labels, model_path, output_model_path, epochs=3):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    train_dataset = ImageDataset(train_image_paths, train_labels, tokenizer, max_length=128)
    val_dataset = ImageDataset(val_image_paths, val_labels, tokenizer, max_length=128)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Training loss: {total_loss / len(train_loader)}")

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = batch
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss = outputs.loss
                total_val_loss += val_loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Validation loss: {total_val_loss / len(val_loader)}")

    model.save_pretrained(output_model_path)
    tokenizer.save_pretrained(output_model_path)

# Paths
train_folder = "C:\\Users\\kesha\\OneDrive\\Desktop\\training"
val_folder = "C:\\Users\\kesha\\OneDrive\\Desktop\\valid"
model_path = "bert-base-uncased"
output_model_path = "C:\\Users\\kesha\\OneDrive\\Desktop\\finetuned_model"

# Create image paths and labels
train_image_paths = [os.path.join(train_folder, fname) for fname in os.listdir(train_folder) if fname.endswith('.jpg')]
train_labels = [0 if "cat" in fname else 1 for fname in os.listdir(train_folder) if fname.endswith('.jpg')]
val_image_paths = [os.path.join(val_folder, fname) for fname in os.listdir(val_folder) if fname.endswith('.jpg')]
val_labels = [0 if "cat" in fname else 1 for fname in os.listdir(val_folder) if fname.endswith('.jpg')]

# Fine-tune the model
fine_tune_bert_model(train_image_paths, train_labels, val_image_paths, val_labels, model_path, output_model_path, epochs=3)
