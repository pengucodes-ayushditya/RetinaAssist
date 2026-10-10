import argparse
import json
import cv2
import numpy as np
import torch
from PIL import Image
from src.pipeline import load_rgb, quality_metrics, crop_retina, tensor_transform
from src.model import load_model


def predict(image_path, checkpoint, override_quality=False):
    rgb=load_rgb(image_path)
    quality=quality_metrics(rgb)
    output={'image':str(image_path),'quality':quality,'disclaimer':'Research prototype only; not a medical diagnosis.'}
    if not quality['usable'] and not override_quality:
        output.update({'status':'rejected_quality','probability_dr':None,'prediction':None})
        return output
    device='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    model=load_model(checkpoint,device)
    pre=tensor_transform(224)(Image.fromarray(crop_retina(rgb))).unsqueeze(0).to(device)
    with torch.no_grad(): prob=float(torch.softmax(model(pre),-1)[0,1].item())
    output.update({'status':'screened','probability_dr':prob,'prediction':'DR suspected — refer for clinical assessment' if prob>=.5 else 'No DR detected by model — not a clinical clearance'})
    return output


def main():
    p=argparse.ArgumentParser(); p.add_argument('--image',required=True); p.add_argument('--checkpoint',default='models/best.pt');p.add_argument('--override-quality',action='store_true');args=p.parse_args()
    print(json.dumps(predict(args.image,args.checkpoint,args.override_quality),indent=2))

if __name__=='__main__':main()
