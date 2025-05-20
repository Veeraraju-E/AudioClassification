import torch
import torch.nn as nn
import os
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append('../')
sys.path.append('../../')

from data.dataset import AudioDataset
from model import AudioClassifierTimeDomain
from utils import evaluate 

BATCH_SIZE = 64
LR = 3e-4
ROOT_DIR = "../audio"
MODEL = AudioClassifierTimeDomain(num_classes=50)
OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=LR)
CRITERION = nn.CrossEntropyLoss()

train_data = AudioDataset(os.path.join(ROOT_DIR,"train"))
val_data = AudioDataset(os.path.join(ROOT_DIR,"valid"))
test_data = AudioDataset(os.path.join(ROOT_DIR,"test"))

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

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
            # print(input_audio.shape, labels.shape) # [B, 1, 220500] and [B]
            
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


train(MODEL, OPTIMIZER, train_loader, val_loader, 25)
test(MODEL, test_loader, model_path='checkpoints/model.pth')