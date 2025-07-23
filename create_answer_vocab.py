import json
import collections
from Levenshtein import distance as lev_distance

# === Editable parameter ===
FREQ_THRESHOLD = 0  # change this as needed

def load_annotations(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def clean_answer(ans):
    return ans.strip().lower()

# Load data
train_data = load_annotations("train.json")
val_data = load_annotations("val.json")
full_data = train_data + val_data

# Global frequency counter
global_counter = collections.Counter()

# Per-question best answers
selected_answers = []

for entry in full_data:
    # Clean all answers
    cleaned_answers = [clean_answer(ans['answer']) for ans in entry['answers']]
    
    # Count frequency in current question
    local_counter = collections.Counter(cleaned_answers)
    
    # Step 1: Most frequent answers in this question
    max_freq = max(local_counter.values())
    candidates = [ans for ans, freq in local_counter.items() if freq == max_freq]
    
    if len(candidates) == 1:
        best = candidates[0]
    else:
        # Step 2: Choose one with highest global frequency
        global_freqs = {ans: global_counter[ans] for ans in candidates}
        max_global_freq = max(global_freqs.values())
        global_candidates = [ans for ans in candidates if global_counter[ans] == max_global_freq]
        
        if len(global_candidates) == 1:
            best = global_candidates[0]
        else:
            # Step 3: Levenshtein distance-based tie-break
            min_total_distance = float('inf')
            best = global_candidates[0]
            for cand in global_candidates:
                total_dist = sum(lev_distance(cand, other) for other in global_candidates if other != cand)
                if total_dist < min_total_distance:
                    min_total_distance = total_dist
                    best = cand

    selected_answers.append(best)
    global_counter.update(cleaned_answers)  # Update global counts after using them

# Final vocabulary with FREQ_THRESHOLD
answer_counter = collections.Counter(selected_answers)
answer_vocab = [ans for ans, freq in answer_counter.items() if freq > FREQ_THRESHOLD]
ans2idx = {ans: idx for idx, ans in enumerate(answer_vocab)}

answer_vocab = ['<unk>'] + answer_vocab
ans2idx['<unk>'] = 5589
