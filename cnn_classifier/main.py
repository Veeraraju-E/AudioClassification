import torch
import torch.nn as nn
import os
import sys
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import models, transforms as T

sys.path.append('../')
sys.path.append('../../')

from data.dataset import AudioDataset, AudioDatasetSpectogram, precompute_spectrograms
from model import AudioClassifierTimeDomain
from utils import evaluate 

CRITERION = nn.CrossEntropyLoss()

def parse_args():
    parser = argparse.ArgumentParser(description='Audio Classification Training')
    parser.add_argument('--model_type', type=str, default='freq', choices=['time', 'freq'], help='Model type')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--root_dir', type=str, default="../audio")
    return parser.parse_args()

def train(
    model,
    optimizer,
    train_dataloader,
    val_dataloader,
    epochs
):
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(
            enumerate(train_dataloader), 
            total=len(train_dataloader), 
            desc=f'Epoch [{epoch+1}/{epochs}]',
            )
        
        for _, data in loop:
            optimizer.zero_grad()
            input_audio, labels = data
            # print(input_audio.shape, labels.shape) # [B, 1, 220500] and [B] for time-domain
            
            preds = model(input_audio)   # [B, num_classes]
            loss = CRITERION(preds, labels)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            loop.set_postfix(loss=loss.item())
        
        epoch_loss = running_loss / len(train_dataloader)
        
        model.eval()
        # Validation
        val_loss, val_acc, val_class_acc = evaluate(model, val_dataloader, CRITERION)
        
        print(f'\nEpoch [{epoch+1}/{epochs}] - Training Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}')
        print("Class-wise val accuracy:")
        for cls, acc in sorted(val_class_acc.items()):
            print(f"Class {cls}: {acc:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'checkpoints/best_model_loss.pth')

def test(
    model,
    test_dataloader,
    model_path,
):
    if model_path and os.path.exists(model_path):
        print(f"using model: {model_path}")
        model.load_state_dict(torch.load(model_path))
    
    test_loss, test_acc, test_class_acc = evaluate(model, test_dataloader, CRITERION)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("Class-wise Test Accuracy:")
    
    for cls, acc in sorted(test_class_acc.items()):
        print(f"Class {cls}: {acc:.4f}")

def main():
    args = parse_args()
    
    BATCH_SIZE = args.batch_size
    LR = args.lr
    ROOT_DIR = args.root_dir
    
    if args.model_type == 'time':
        MODEL = AudioClassifierTimeDomain(num_classes=50)
        dataset_class = AudioDataset
        transforms = None
    else:
        MODEL = models.resnet101(pretrained=True)
        for param in MODEL.parameters():
            param.requires_grad = False
        MODEL.fc = nn.Sequential(
            nn.Linear(MODEL.fc.in_features, 500),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(500, 50)  # ESC50
        )
        dataset_class = AudioDatasetSpectogram
        # precompute spectrograms if using freq domain -> do it once
        # for split in ['train', 'valid', 'test']:
        #     precompute_spectrograms(os.path.join(ROOT_DIR, split))
        transforms = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.Resize((224, 224)),  # for resnet inputs
            T.ToTensor(),
        ])
    
    OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=LR)

    # Data loading
    train_data = dataset_class(os.path.join(ROOT_DIR, "train"), transforms=transforms)
    val_data = dataset_class(os.path.join(ROOT_DIR, "valid"), transforms=transforms)
    test_data = dataset_class(os.path.join(ROOT_DIR, "test"), transforms=transforms)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Train and test
    train(MODEL, OPTIMIZER, train_loader, val_loader, args.epochs)
    test(MODEL, test_loader, model_path='checkpoints/model.pth')

    
if __name__ == "__main__":
    main()
