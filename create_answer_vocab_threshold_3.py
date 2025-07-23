                                                                                                                                                           
import json
import collections

# === Editable parameter ===
FREQ_THRESHOLD = 3  # change this as needed

def load_annotations(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def clean_answer(ans):
    return ans.strip().lower()

# Load data
train_data = load_annotations("train.json")
val_data = load_annotations("val.json")
full_data = train_data + val_data

# Step 1: Collect all cleaned answers globally
all_cleaned_answers = []
for entry in full_data:
    all_cleaned_answers.extend([clean_answer(ans['answer']) for ans in entry['answers']])

# Step 2: Build global frequency counter
global_counter = collections.Counter(all_cleaned_answers)

# Step 3: Filter based on FREQ_THRESHOLD
filtered_answers = [ans for ans, freq in global_counter.items() if freq > FREQ_THRESHOLD]

# Step 4: Build answer_vocab and ans2idx
answer_vocab =  filtered_answers+ ['<unk>']
ans2idx = {ans: idx for idx, ans in enumerate(answer_vocab)}

