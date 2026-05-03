import torch
import torch.nn as nn
from transformers import ASTModel


class CustomAST(nn.Module):
    def __init__(self, num_classes=4, dropout=0.3):
        super().__init__()
        self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, num_classes)
        )

    def forward(self, x):
        outputs = self.ast(x)
        embeddings = outputs.last_hidden_state.mean(dim=1) 
        logits = self.classifier(embeddings)
        
        return logits
