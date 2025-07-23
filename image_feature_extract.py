import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import open_clip

# Load CLIP ViT-B/32
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(device)
model.eval()

# Paths
splits = ['train', 'val', 'test']
json_files = {s: f"{s}.json" for s in splits}
image_dirs = {s: s for s in splits}
feature_dirs = {s: f"{s}_features" for s in splits}

# Create feature directories
for s in splits:
    os.makedirs(feature_dirs[s], exist_ok=True)

# Feature extraction function
def extract_and_save_features(json_file, image_dir, output_dir):
    with open(json_file, 'r') as f:
        data = json.load(f)

    for item in tqdm(data, desc=f"Processing {json_file}"):
        image_name = item['image']
        image_path = os.path.join(image_dir, image_name)
        feature_path = os.path.join(output_dir, image_name.replace('.jpg', '.pt'))

        # Skip if already extracted
        if os.path.exists(feature_path):
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image_input).squeeze(0).cpu()

            torch.save(image_features, feature_path)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

# Run extraction for each split
for split in splits:
    extract_and_save_features(json_files[split], image_dirs[split], feature_dirs[split])
