import torchvision.transforms.v2 as transforms
import torchvision.datasets as datasets

def get_transform(type):
    mean, std = [0.4914, 0.4821, 0.4465], [0.2471, 0.2435, 0.2616]

    if type=="train":
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224),  
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        return transform_train

    transform_test = transforms.Compose([
        transforms.Resize(224), 
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform_test