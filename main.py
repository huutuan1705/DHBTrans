import argparse

if __name__ == "__main__":
    parsers = argparse.ArgumentParser(description='Deep Balanced Hashing based Transformer')
    parsers.add_argument('--dataset_name', type=str, default='cifar10', help="cifar10/nus_wide/image_net")
    
    parsers.add_argument('--batch_size', type=int, default=16)
    parsers.add_argument('--threads', type=int, default=4)
    parsers.add_argument('--lr', type=float, default=1e-3)
    parsers.add_argument('--threads', type=int, default=4)
    parsers.add_argument('--epochs', type=int, default=200)