import torch
import argparse
import torch.utils.data as data

from torch import optim

from dataset.cifa10 import CiFar10_Dataset
from dataset.nus_wide import NUS_WIDE_Dataset
from dataset.image_net import ImageNet_Dataset
from utils.training import training
from models.model import DHBTrans
from losses.dhbtrans_loss import DHBLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloader(args):
    datasets = {
        "cifar10": "CiFar10_Dataset",
        "nus_wide": "NUS_WIDE_Dataset",
        "image_net": "ImageNet_Dataset"
    }
    train_dataset = eval(datasets[args.dataset_name] + "(args, mode='train')")
    query_dataset = eval(datasets[args.dataset_name] + "(args, mode='query')")
    database_dataset = eval(datasets[args.dataset_name] + "(args, mode='database')")
    
    train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=int(args.threads))
    query_dataloader = data.DataLoader(query_dataset, batch_size=args.step_size, shuffle=False, num_workers=int(args.threads))
    db_dataloader = data.DataLoader(database_dataset, batch_size=args.step_size, shuffle=False, num_workers=int(args.threads))
    
    return train_dataloader, query_dataloader, db_dataloader

if __name__ == "__main__":
    parsers = argparse.ArgumentParser(description='Deep Balanced Hashing based Transformer')
    parsers.add_argument('--dataset_name', type=str, default='cifar10', help="cifar10/nus_wide/image_net")
    parsers.add_argument('--bit_size', type=int, default=16, help="16/32/64")
    parsers.add_argument('--gamma', type=float, default=0.01)
    parsers.add_argument('--lamda', type=float, default=0.1)
    parsers.add_argument('--alpha', type=float, default=0.1)
    parsers.add_argument('--q', type=float, default=0.1)
    
    parsers.add_argument('--batch_size', type=int, default=16)
    parsers.add_argument('--step_size', type=int, default=64)
    parsers.add_argument('--threads', type=int, default=4)
    parsers.add_argument('--lr', type=float, default=1e-3)
    parsers.add_argument('--weight_decay', type=float, default=1e-5)
    parsers.add_argument('--seed', type=int, default=42)
    parsers.add_argument('--epochs', type=int, default=200)
    parsers.add_argument('--checkpoint', type=int, default=10)
    parsers.add_argument('--load_pretrained', type=bool, default=False)
    parsers.add_argument('--pretrained_dir', type=str, default="./../")
    
    args = parsers.parse_args()
    
    train_dataloader, query_dataloader, db_dataloader = get_dataloader(args)
    model = DHBTrans(args).to(device)
    dhb_loss = DHBLoss(args).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.load_pretrained:
        model.load_state_dict(torch.load(args.pretrained_dir))
    
    training(model, train_dataloader, query_dataloader, db_dataloader, optimizer, dhb_loss, args)