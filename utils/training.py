import torch
import numpy as np
from tqdm import tqdm

from models.model import DHBTrans
from losses.dhbtrans_loss import total_loss
from utils.hamming import compute_similarity_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def training(train_dataloader, optimizer, args):
    model = DHBTrans(args).to(device)
    
    for epoch in range(args.epochs):
        print(f"Epoch: {epoch+1} / {args.epochs}")
        
        for _, batch_data in enumerate(tqdm(train_dataloader)):
            model.train()
            optimizer.zero_grad()
            
            images = batch_data[0]
            labels = batch_data[1]
            output = model(images)
            similarity = compute_similarity_matrix(labels)
        
            loss = total_loss(output, similarity, args)
            
            loss.backward()
            optimizer.step()
            
        