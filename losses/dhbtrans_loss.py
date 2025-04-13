import torch
import torch.nn as nn
from utils.hamming import compute_inner_product

class DHBLoss(nn.Module):
    def __init__(self, args, h_matrix, similarity):
        super(DHBLoss, self).__init__()
            
    def ce_loss(self, h_matrix, similarity):
        omega = compute_inner_product(h_matrix)
        log_part = torch.log1p(torch.exp(omega))
        return log_part - similarity * omega

    def mse_loss(self, h_matrix, similarity, q):
        omega = compute_inner_product(h_matrix)
        diff = (omega + q) / 2 - similarity * q
        loss = torch.pow(diff, 2)
        
        return loss

    def quantization_loss(self, h_matrix):
        return torch.sum(torch.abs(torch.abs(h_matrix) - 1.0))

    def hash_balance_loss(self, h_matrix):
        bit_means = h_matrix.mean(dim=0)
        loss = torch.sum(bit_means ** 2)  # ||mean||^2
        return loss

    def forward(self, h_matrix, similarity, args):
        sigma = args.sigma
        lamda = args.lamda 
        alpha = args.alpha 
        
        loss = self.ce_loss(h_matrix, similarity) + sigma*self.mse_loss(h_matrix, similarity, args.q) \
            + lamda*self.quantization_loss(h_matrix) + alpha*self.hash_balance_loss(h_matrix)
            
        return loss