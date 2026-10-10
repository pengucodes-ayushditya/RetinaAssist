"""Recreate the held-out APTOS split and evaluate a saved checkpoint."""
import argparse
import json
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.pipeline import load_labels, RetinaDataset
from src.model import load_model
from src.train import evaluate_loader


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv',required=True);p.add_argument('--image-root',required=True)
    p.add_argument('--checkpoint',default='models/best.pt')
    p.add_argument('--batch-size',type=int,default=16)
    args=p.parse_args()
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    seed=saved.get('seed',42)
    df=load_labels(args.csv,args.image_root)
    _,test=train_test_split(df,test_size=.15,stratify=df.label,random_state=seed)
    loader=DataLoader(RetinaDataset(test,size=saved.get('size',224)),batch_size=args.batch_size)
    device='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(json.dumps(evaluate_loader(load_model(args.checkpoint,device),loader,device),indent=2))

if __name__=='__main__':main()
