"""Train on APTOS labels with stratified train/val/test splits."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.pipeline import load_labels, RetinaDataset
from src.model import make_model
from src.metrics import calculate_metrics


def evaluate_loader(model, loader, device):
    model.eval()
    labels, scores = [], []
    with torch.no_grad():
        for images, target in loader:
            output = model(images.to(device))
            scores += torch.softmax(output, -1)[:, 1].cpu().tolist()
            labels += target.tolist()
    return calculate_metrics(labels, scores)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--image-root', required=True)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--size', type=int, default=224)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default='models/best.pt')
    p.add_argument('--pretrained', action='store_true', help='Download ImageNet weights; recommended for real training')
    p.add_argument('--num-workers', type=int, default=0)
    args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    frame = load_labels(args.csv, args.image_root)
    if min(frame.label.value_counts()) < 3:
        raise ValueError('Need at least 3 images of each class for stratified train/validation/test split')
    first, test = train_test_split(frame, test_size=.15, stratify=frame.label, random_state=args.seed)
    train, val = train_test_split(first, test_size=.17647, stratify=first.label, random_state=args.seed)
    loaders = {name: DataLoader(RetinaDataset(split, train=(name=='train'), size=args.size), batch_size=args.batch_size, shuffle=(name=='train'), num_workers=args.num_workers) for name, split in [('train',train),('val',val),('test',test)]}
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    model = make_model(args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1. / max(1, (train.label==c).sum()) for c in (0,1)], device=device, dtype=torch.float))
    best = -1
    output = Path(args.out); output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train(); losses=[]
        for images, targets in loaders['train']:
            optimizer.zero_grad(); logits=model(images.to(device)); loss=loss_fn(logits, targets.to(device)); loss.backward(); optimizer.step(); losses.append(loss.item())
        result=evaluate_loader(model, loaders['val'], device)
        score=result['roc_auc'] if result['roc_auc'] is not None else result['f1']
        print(f'Epoch {epoch+1}/{args.epochs} loss={np.mean(losses):.4f} val_auc={result["roc_auc"]} val_f1={result["f1"]:.4f}', flush=True)
        if score > best:
            best=score
            torch.save({'model': model.state_dict(), 'size': args.size, 'seed': args.seed, 'val_metrics': result}, output)
    saved=torch.load(output,map_location='cpu',weights_only=True)
    model.load_state_dict(saved['model'])
    test_metrics=evaluate_loader(model,loaders['test'],device)
    result={'train_count':len(train),'val_count':len(val),'test_count':len(test),'seed':args.seed,'validation':saved['val_metrics'],'test':test_metrics}
    report=output.with_suffix('.metrics.json'); report.write_text(json.dumps(result,indent=2))
    print(f'Saved checkpoint: {output}\nSaved test report: {report}\nTest metrics: {test_metrics}')

if __name__=='__main__': main()
