import os
from PIL import Image
import shutil
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

# Define paths
yolo_dataset_path = "split_dataset"
output_dataset_path = "clip_dataset"
hf_repo_name = "AtAndDev/clip-bicycle-e-bike"

# Create output directories
os.makedirs(os.path.join(output_dataset_path, "train"), exist_ok=True)
os.makedirs(os.path.join(output_dataset_path, "val"), exist_ok=True)

# Define class mapping
class_mapping = {0: "bicycle", 1: "e-bike"}

def convert_yolo_to_classification(yolo_path, output_path, split):
    images = []
    labels = []
    
    # Iterate through images in the split folder
    image_folder = os.path.join(yolo_path, split, "images")
    label_folder = os.path.join(yolo_path, split, "labels")
    
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        label_path = os.path.join(label_folder, image_name.replace(".jpg", ".txt"))
        
        # Check if label file exists
        if not os.path.exists(label_path):
            continue
        
        # Read the label file
        with open(label_path, "r") as f:
            lines = f.readlines()
        
        # Check if the image contains class 0 or 1
        for line in lines:
            class_id = int(line.split()[0])
            if class_id in class_mapping:
                # Load the image
                image = Image.open(image_path)
                images.append(image)
                labels.append(class_mapping[class_id])
                break
    
    # Create a Hugging Face Dataset
    dataset = Dataset.from_dict({"image": images, "label": labels})
    return dataset

# Convert train and val splits
train_dataset = convert_yolo_to_classification(yolo_dataset_path, output_dataset_path, "train")
val_dataset = convert_yolo_to_classification(yolo_dataset_path, output_dataset_path, "val")

# Create a DatasetDict
dataset_dict = DatasetDict({"train": train_dataset, "val": val_dataset})

# Push to Hugging Face Hub
dataset_dict.save_to_disk(hf_repo_name)