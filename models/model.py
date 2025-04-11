import torch
import torch.nn as nn

from timm.models.vision_transformer import VisionTransformer

class DHBTrans(nn.Module):
    def __init__(self):
        super(DHBTrans, self).__init__()
        