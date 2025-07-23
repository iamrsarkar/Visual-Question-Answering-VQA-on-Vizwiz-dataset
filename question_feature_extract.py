import os
import json
import torch
from tqdm import tqdm
import open_clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, _ = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='openai')
model = model.to(device)
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Paths
splits = ['train', 'val', 'test']

json_files = {s: f"{s}.json" for s in splits}
question_feature_dirs = {s: f"{s}_questions" for s in splits}  # New directories

# Create question feature directories
for s in splits:
    os.makedirs(question_feature_dirs[s], exist_ok=True)

def extract_and_save_text_features(json_file, output_dir):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    for item in tqdm(data, desc=f"Processing {json_file}"):
        question = item['question']
        feature_path = os.path.join(output_dir, f"text_{item['image'].replace('.jpg', '.pt')}")  # Same filename as image
        
        if os.path.exists(feature_path):
            continue
        
        try:
            with torch.no_grad():
                text_input = tokenizer([question]).to(device)
                text_features = model.encode_text(text_input).squeeze(0).cpu()
            torch.save(text_features, feature_path)
        except Exception as e:
            print(f"Error processing question: {question}\nError: {e}")

# Run extraction
for split in splits:
    extract_and_save_text_features(json_files[split], question_feature_dirs[split])
