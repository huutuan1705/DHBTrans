import torch
from tqdm import tqdm

from utils.hamming import compute_similarity_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_result(model, dataloader):
    model.eval()
    outputs, labels = [], []
    with torch.no_grad():
        for _, batch_data in enumerate(tqdm(dataloader)):
            images = batch_data[0].to(device)
            label = batch_data[1].to(device)
            output = model(images)
            
            outputs.append(output)
            labels.append(label)
    
    return torch.sign(torch.cat(outputs)), torch.cat(labels)

def evaluate(model, query_dataloader, db_dataloader):
    model.eval()
    query_binaries, query_labels = compute_result(model, query_dataloader)
    db_binaries, db_labels = compute_result(model, db_dataloader)
    
    AP = []
    db_size = torch.arange(1, db_binaries.size(0) + 1)
    for i in range(query_binaries.size(0)):
        query_label, query_binary = query_labels[i], query_binaries[i]
        _, query_result = torch.sum((query_binary != db_binaries).long(), dim=1).sort()
        correct = (query_label == db_labels[query_result].to(device)).float()
        precision = torch.cumsum(correct, dim=0) / db_size
        AP.append(torch.sum(precision*correct) / torch.sum(correct))
    
    mAP = torch.mean(AP)
    return mAP

def training(model, train_dataloader, query_dataloader, db_dataloader, optimizer, dhb_loss, args):
    best_mAP = 0
    for epoch in range(args.epochs):
        print(f"Epoch: {epoch+1} / {args.epochs}")
        mean_loss = 0
        count = 0
        # for _, batch_data in enumerate(tqdm(train_dataloader)):
        #     model.train()
        #     optimizer.zero_grad()
            
        #     images = batch_data[0].to(device)
        #     labels = batch_data[1]
        #     output = model(images)
        #     similarity = compute_similarity_matrix(labels)
        
        #     loss = dhb_loss(output, similarity)
        #     mean_loss += loss.item()
        #     loss.backward()
        #     optimizer.step()
        #     count += 1
            
        # mean_loss = mean_loss / count   
        # print('Training loss: {:.4f}'.format(mean_loss)) 
        
        if epoch % args.checkpoint == 0:
            mAP = evaluate(model, query_dataloader, db_dataloader)
            print('Mean Average Precision(mAP): {:.4f}'.format(mAP)) 
            if mAP >= best_mAP:
                best_name = "best_" + args.dataset_name + ".pth"
                torch.save(model.state_dict(), best_name)
            
            last_name = "last_" + args.dataset_name + ".pth"
            torch.save(model.state_dict(), last_name)