import numpy as np
from sklearn.metrics import accuracy_score
import torch
from tqdm import tqdm

def compute_class_accuracy(all_preds, all_labels):
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    overall_accuracy = accuracy_score(all_labels, all_preds)
    
    class_accuracy = {}
    unique_classes = np.unique(all_labels)
    
    for cls in unique_classes:
        cls_indices = (all_labels == cls)
        if np.sum(cls_indices) > 0:
            cls_accuracy = np.mean(all_preds[cls_indices] == cls)
            class_accuracy[int(cls)] = float(cls_accuracy)
    
    return overall_accuracy, class_accuracy

def evaluate(model, dataloader, criterion):

    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Evaluating"):
            input_audio, labels = data
            outputs = model(input_audio)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = running_loss / len(dataloader)
    overall_accuracy, class_accuracy = compute_class_accuracy(all_preds, all_labels)
    
    return avg_loss, overall_accuracy, class_accuracy