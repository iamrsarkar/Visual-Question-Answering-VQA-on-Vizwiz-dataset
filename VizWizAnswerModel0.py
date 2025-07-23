import torch
import torch.nn as nn

class VizWizAnswerModel(nn.Module):
    def __init__(self, num_answers):
        super().__init__()

        # Directly predict from 1024D input
        self.answer_head = nn.Sequential(
            nn.Linear(1024, num_answers),
            nn.Softmax(dim=1)  # For soft cross-entropy loss
        )

    def forward(self, image_features, question_features):
        # Concatenate features (512D image + 512D text = 1024D)
        combined = torch.cat([image_features, question_features], dim=1)

        # Direct prediction
        answer_probs = self.answer_head(combined)

        return answer_probs
