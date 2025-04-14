import torch
import torch.nn as nn
from utils.hamming import compute_inner_product

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DHBLoss(nn.Module):
    def __init__(self, args):
        super(DHBLoss, self).__init__()
        self.args = args
            
    def ce_loss(self, h_matrix, similarity):
        omega = compute_inner_product(h_matrix)
        log_part = torch.log1p(torch.exp(omega))
        return torch.mean(log_part - similarity * omega)

    def mse_loss(self, h_matrix, similarity, q):
        q = h_matrix.shape[1]
        omega = compute_inner_product(h_matrix)
        diff = ((omega + q) / 2 - similarity * q) ** 2
        loss = torch.mean(diff)
        
        return loss

    def quantization_loss(self, h_matrix):
        return torch.mean(torch.abs(torch.abs(h_matrix) - 1.0))

    def hash_balance_loss(self, h_matrix):
        bit_means = torch.mean(h_matrix, dim=0)
        loss = 4 * torch.sum(bit_means ** 2)  # ||mean||^2
        return loss

    def forward(self, h_matrix, similarity):
        gamma = torch.tensor(self.args.gamma, device=device)
        lamda = torch.tensor(self.args.lamda, device=device) 
        alpha = torch.tensor(self.args.alpha, device=device) 
        h_matrix = h_matrix.to(device)
        similarity = similarity.to(device)
        
        loss = self.ce_loss(h_matrix, similarity) + gamma*self.mse_loss(h_matrix, similarity, self.args.q) \
            + lamda*self.quantization_loss(h_matrix) + alpha*self.hash_balance_loss(h_matrix)
            
        return loss