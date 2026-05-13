#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
تحسين Bi-LSTM: إصلاح المشكلة المتحيزة
Improved Bi-LSTM: Fixing the Bias Issue
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from collections import Counter
import re
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# =============================================================
# 1. المعاملات المحسّنة
# =============================================================
EMBEDDING_DIM = 200      # مخفّض من 256 لتجنب الإفراط
HIDDEN_SIZE = 256        # مخفّض من 512 للاستقرار
NUM_LAYERS = 2           # مخفّض من 4 لتجنب الإفراط
DROPOUT = 0.3            # مخفّض من 0.5 للتعلم الأفضل
MAX_VOCAB_SIZE = 20000   # مخفّض من 30000
MAX_SEQ_LEN = 128        # مخفّض من 256 للكفاءة

BATCH_SIZE = 32          # مخفّض من 64
EPOCHS = 15
LEARNING_RATE = 5e-4     # مخفّض من 1e-3 للاستقرار
WEIGHT_DECAY = 1e-5      # تنظيم L2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")
print(f"معاملات محسّنة - Optimized Parameters:")
print(f"  Embedding: {EMBEDDING_DIM}, Hidden: {HIDDEN_SIZE}")
print(f"  Layers: {NUM_LAYERS}, Dropout: {DROPOUT}")
print(f"  Learning Rate: {LEARNING_RATE}, Batch Size: {BATCH_SIZE}\n")

# =============================================================
# 2. دوال المعالجة والفئات
# =============================================================

def preprocess_text(text):
    """معالجة النص"""
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zçşğüöı\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class Vocabulary:
    def __init__(self, max_size):
        self.max_size = max_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def build(self, texts):
        word_counts = Counter()
        for text in texts:
            cleaned = preprocess_text(text)
            word_counts.update(cleaned.split())
        most_common = word_counts.most_common(self.max_size - 2)
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        print(f"[Vocab] {len(self.word2idx)} كلمة")

    def encode(self, text):
        cleaned = preprocess_text(text)
        tokens = cleaned.split()
        return [self.word2idx.get(word, 1) for word in tokens]

class NewsDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts, self.labels, self.vocab, self.max_len = texts, labels, vocab, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.vocab.encode(self.texts[idx])
        encoded = (encoded + [0] * self.max_len)[:self.max_len]
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

class ImprovedBiLSTMClassifier(nn.Module):
    """نموذج محسّن مع آلية الانتباه"""

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # آلية الانتباه
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x, apply_sigmoid=False):
        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)

        # LSTM
        lstm_out, (hidden, _) = self.lstm(embedded)  # (batch, seq_len, hidden_size*2)

        # Attention
        attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = self.softmax(attention_weights)  # (batch, seq_len, 1)

        # Weighted sum
        context = (lstm_out * attention_weights).sum(dim=1)  # (batch, hidden_size*2)

        # FC layers
        context = self.dropout(context)
        hidden = self.relu(self.fc1(context))
        hidden = self.dropout(hidden)
        output = self.fc2(hidden)

        if apply_sigmoid:
            output = torch.sigmoid(output)

        return output.squeeze(1)

# =============================================================
# 3. دالة التدريب
# =============================================================

def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for i, (batch_texts, batch_labels) in enumerate(loader):
        batch_texts, batch_labels = batch_texts.to(DEVICE), batch_labels.to(DEVICE)

        predictions = model(batch_texts)
        loss = criterion(predictions, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        probs = torch.sigmoid(predictions)
        all_preds.extend((probs > 0.5).cpu().detach().numpy())
        all_labels.extend(batch_labels.cpu().numpy())

        if (i + 1) % 50 == 0:
            acc = accuracy_score(all_labels, all_preds)
            print(f"   [Epoch {epoch+1}] Batch {i+1}/{len(loader)} | Loss: {loss.item():.4f} | Acc: {acc*100:.1f}%")

    return total_loss / len(loader), accuracy_score(all_labels, all_preds)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_texts, batch_labels in loader:
            batch_texts, batch_labels = batch_texts.to(DEVICE), batch_labels.to(DEVICE)
            predictions = model(batch_texts)
            loss = criterion(predictions, batch_labels)
            total_loss += loss.item()
            probs = torch.sigmoid(predictions)
            all_preds.extend((probs > 0.5).cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

    return total_loss / len(loader), acc, precision, recall, f1

# =============================================================
# 4. البرنامج الرئيسي
# =============================================================

def main():
    print("\n" + "="*80)
    print("محسّن Bi-LSTM مع آلية الانتباه - Improved Bi-LSTM with Attention")
    print("="*80)

    # تحميل البيانات
    file_path = os.path.join("data", "processed", "combined_dataset.csv")
    if not os.path.exists(file_path):
        file_path = os.path.join("data", "processed", "WELFake_cleaned.csv")

    print(f"\n[Data] تحميل: {file_path}")
    df = pd.read_csv(file_path).dropna(subset=['content', 'label'])

    # فحص التوازن
    print(f"\nتوزيع الفئات:")
    print(f"  Fake (1): {int(df['label'].sum())} ({df['label'].mean()*100:.1f}%)")
    print(f"  Real (0): {len(df) - int(df['label'].sum())} ({(1-df['label'].mean())*100:.1f}%)")

    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(
        df['content'].tolist(),
        df['label'].tolist(),
        test_size=0.2,
        stratify=df['label'],
        random_state=42
    )

    print(f"\nتقسيم البيانات:")
    print(f"  Training: {len(X_train)}")
    print(f"  Testing: {len(X_test)}")

    # Vocab
    vocab = Vocabulary(MAX_VOCAB_SIZE)
    vocab.build(X_train)

    # DataLoaders
    train_loader = DataLoader(
        NewsDataset(X_train, y_train, vocab, MAX_SEQ_LEN),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    test_loader = DataLoader(
        NewsDataset(X_test, y_test, vocab, MAX_SEQ_LEN),
        batch_size=BATCH_SIZE
    )

    # Model
    model = ImprovedBiLSTMClassifier(
        len(vocab.word2idx),
        EMBEDDING_DIM,
        HIDDEN_SIZE,
        NUM_LAYERS,
        DROPOUT
    ).to(DEVICE)

    # Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )

    # Early Stopping
    PATIENCE = 4
    best_f1 = 0
    patience_counter = 0

    print("\n" + "="*80)
    print("التدريب - Training")
    print("="*80)

    for epoch in range(EPOCHS):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        v_loss, v_acc, v_prec, v_rec, v_f1 = evaluate(model, test_loader, criterion)

        print(f"\nEpoch {epoch+1}:")
        print(f"  Train Loss: {t_loss:.4f}, Acc: {t_acc*100:.2f}%")
        print(f"  Test  Loss: {v_loss:.4f}, Acc: {v_acc*100:.2f}%")
        print(f"  Precision: {v_prec:.4f}, Recall: {v_rec:.4f}, F1: {v_f1:.4f}")

        # Save best model
        if v_f1 > best_f1:
            best_f1 = v_f1
            patience_counter = 0
            save_path = os.path.join("models", "saved", "bilstm_model_improved.pt")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_word2idx': vocab.word2idx,
                'hyperparameters': {
                    'embedding_dim': EMBEDDING_DIM,
                    'hidden_size': HIDDEN_SIZE,
                    'num_layers': NUM_LAYERS,
                    'dropout': DROPOUT,
                    'max_seq_len': MAX_SEQ_LEN,
                    'vocab_size': len(vocab.word2idx)
                }
            }, save_path)
            print(f"  ✓ تم حفظ النموذج (F1: {v_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  ℹ Early Stopping: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("  ✗ توقيف التدريب")
                break

        scheduler.step(v_f1)

    print("\n" + "="*80)
    print(f"✓ اكتمل التدريب! أفضل F1: {best_f1:.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
