import torch
import torch.nn as nn
from transformers import ViTModel

class DHBTrans(nn.Module):
    def __init__(self, args):
        super(DHBTrans, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-large-patch16-224-in21k")
        self.relu = nn.ReLU()
        self.linear = nn.Linear(1024, args.bit_size)
        
    def forward(self, x):
        outputs = self.vit(x)
        x = outputs.last_hidden_state[:, 0, :]  # <-- get represented feature
        x = self.relu(x)
        x = self.linear(x)
        return x