import cv2
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from PIL import Image

def preprocess_sample_image(sample_image_path, model_path, output_image_path):
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    image = cv2.imread(sample_image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {sample_image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use adaptive thresholding to binarize the image
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 15)

    # Detect and remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detect_horizontal = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    contours, _ = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 50 and h < 10:  # Filter for horizontal lines
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), -1)  # Remove line

    # Detect and remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detect_vertical = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    contours, _ = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 10 and h > 50:  # Filter for vertical lines
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), -1)  # Remove line

    cv2.imwrite(output_image_path, image)
    print(f"Processed image saved to: {output_image_path}")

# Paths
sample_image_path = r"C:\Users\kesha\OneDrive\Desktop\test2.jpg"
output_image_path = r"C:\Users\kesha\OneDrive\Desktop\check5.png"
model_path = "C:\\Users\\kesha\\OneDrive\\Desktop\\finetuned_model"

# Preprocess the sample image
preprocess_sample_image(sample_image_path, model_path, output_image_path)
