import torch
import torch.nn as nn

class VizWizAnswerModel(nn.Module):
    def __init__(self, num_answers):
        super().__init__()
        
        # Feature fusion 
        self.fusion = nn.Sequential(
            nn.Linear(1024, 512),  # Concatenated image+text features
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5), 
        )
        
        # Answer prediction head
        self.answer_head = nn.Sequential(
            nn.Linear(512, num_answers),
            nn.Softmax(dim=1)  # For soft cross-entropy loss
        )
     
    def forward(self, image_features, question_features):
        # Concatenate features (512D image + 512D text = 1024D)
        combined = torch.cat([image_features, question_features], dim=1)
        
        # Fusion layers
        fused = self.fusion(combined)
        
        # Answer probabilities
        answer_probs = self.answer_head(fused)
        
        return answer_probs

