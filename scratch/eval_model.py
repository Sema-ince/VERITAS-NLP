import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import sys
import os

# Ad hoc import to use the classes defined in train_bilstm.py
sys.path.append(os.getcwd())
from scripts.train_bilstm import BiLSTMClassifier, NewsDataset, Vocabulary

def evaluate_current_model():
    model_path = 'models/saved/bilstm_model.pt'
    if not os.path.exists(model_path):
        print("Model not found!")
        return

    checkpoint = torch.load(model_path, map_location='cpu')
    hp = checkpoint['hyperparameters']
    vocab_dict = checkpoint['vocab_word2idx']

    model = BiLSTMClassifier(hp['vocab_size'], hp['embedding_dim'], hp['hidden_size'], hp['num_layers'], hp['dropout'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    df = pd.read_csv('data/processed/combined_dataset.csv').sample(2000, random_state=42)
    vocab = Vocabulary(30000)
    vocab.word2idx = vocab_dict
    
    dataset = NewsDataset(df['content'].tolist(), df['label'].tolist(), vocab, hp['max_seq_len'])
    loader = DataLoader(dataset, batch_size=32)
    
    criterion = nn.BCEWithLogitsLoss()
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for texts, labels in loader:
            outputs = model(texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    print(f"Final Metrics for the Current Model:")
    print(f"Loss: {total_loss/len(loader):.4f}")
    print(f"Accuracy: {correct/total:.4f}")

if __name__ == "__main__":
    evaluate_current_model()
