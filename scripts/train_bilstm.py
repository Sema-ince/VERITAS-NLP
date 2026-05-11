"""
VERITAS-NLP: Bi-LSTM Full Training Dashboard
==============================================
Bu script, 90,000+ haber uzerinde dengeli ve dogru bir egitim yapmak icin tasarlanmistir.
Tum bias sorunlari giderilmistir.

[ISLAH - DUZELTMELER]:
1. preprocess_text() fonksiyonu eklendi - egitim ve uygulama arasindan tutarsizlik giderildi
2. EPOCHS 3'ten 10'a cikarildi
3. Vocab.build() ve encode() artik temizlenmis metin kullaniyor
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
import re

# =============================================================
# 1. PARAMETRELER
# =============================================================
EMBEDDING_DIM = 128
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.5
MAX_VOCAB_SIZE = 30000
MAX_SEQ_LEN = 256

BATCH_SIZE = 64
EPOCHS = 10               # ✅ ISLAH: 3 -> 10
LEARNING_RATE = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================
# ✅ ISLAH: ORTAK ONISLEME FONKSİYONU
# Bu fonksiyon hem egitimde hem app.py'de ayni sekilde kullanilmalidir.
# =============================================================
def preprocess_text(text):
    """Metni temizler: kucuk harf, sadece harf karakterleri, fazla bosluk kaldirilir."""
    if not text:
        return ""
    text = str(text).lower()
    # Sadece Ingilizce + Turkce harfleri koru
    text = re.sub(r'[^a-zçşğüöı\s]', ' ', text)
    # Fazla boslukları temizle
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# =============================================================
# 2. YARDIMCI SINIFLAR
# =============================================================
class Vocabulary:
    def __init__(self, max_size):
        self.max_size = max_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
    
    def build(self, texts):
        word_counts = Counter()
        for text in texts:
            # ✅ ISLAH: onislenmis metin uzerinden sozluk olustur
            cleaned = preprocess_text(text)
            word_counts.update(cleaned.split())
        most_common = word_counts.most_common(self.max_size - 2)
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        print(f"[Vocab] Sozluk olusturuldu: {len(self.word2idx)} kelime")
    
    def encode(self, text):
        # ✅ ISLAH: onislenmis metin uzerinden encode et
        cleaned = preprocess_text(text)
        tokens = cleaned.split()
        return [self.word2idx.get(word, 1) for word in tokens]

class NewsDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts, self.labels, self.vocab, self.max_len = texts, labels, vocab, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        encoded = self.vocab.encode(self.texts[idx])
        encoded = (encoded + [0] * self.max_len)[:self.max_len]
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, apply_sigmoid=False):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        combined = torch.cat((hidden[-2], hidden[-1]), dim=1)
        combined = self.dropout(combined)
        output = self.fc(combined)
        if apply_sigmoid: output = self.sigmoid(output)
        return output.squeeze(1)

# =============================================================
# 3. EGITIM DONGUSU
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
        optimizer.step()
        
        total_loss += loss.item()
        probs = torch.sigmoid(predictions)
        all_preds.extend((probs > 0.5).cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())
        
        if (i + 1) % 50 == 0:
            acc = accuracy_score(all_labels, all_preds)
            print(f"   [Epoch {epoch+1}] Batch {i+1}/{len(loader)} | Kayip: {loss.item():.4f} | Acc: %{acc*100:.1f}")
            
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
    return total_loss / len(loader), accuracy_score(all_labels, all_preds)

# =============================================================
# 4. ANA PROGRAM
# =============================================================
def main():
    print("\n" + "="*60)
    print(" VERITAS-NLP: TAM KAPSAMLI EGITIM BASLIYOR")
    print("="*60)
    print(f"[Device] Cihaz: {DEVICE}")

    # Veri Yukleme
    file_path = os.path.join("data", "processed", "combined_dataset.csv")
    if not os.path.exists(file_path): file_path = os.path.join("data", "processed", "WELFake_cleaned.csv")
    
    print(f"[Data] Veri yukleniyor: {file_path}")
    df = pd.read_csv(file_path).dropna(subset=['content', 'label'])
    
    # Turkce veri dengeleme (Oversampling 4x)
    print("[Process] Dil tespiti ve dengeleme yapiliyor...")
    if 'language' not in df.columns:
        df['language'] = df['content'].apply(lambda x: 'tr' if re.search(r'[çşğüöı]', str(x).lower()) else 'en')
    
    tr_data = df[df['language'] == 'tr']
    en_data = df[df['language'] == 'en']
    print(f"   - Orijinal: EN={len(en_data):,}, TR={len(tr_data):,}")
    
    tr_oversampled = pd.concat([tr_data] * 4, ignore_index=True)
    df = pd.concat([en_data, tr_oversampled]).sample(frac=1).reset_index(drop=True)
    print(f"   - Dengeli Toplam: {len(df):,}")

    # Veri Bolme
    X_train, X_test, y_train, y_test = train_test_split(df['content'].tolist(), df['label'].tolist(), test_size=0.2, stratify=df['label'], random_state=42)
    
    # Vocab
    vocab = Vocabulary(MAX_VOCAB_SIZE)
    vocab.build(X_train)
    
    # Loaders
    train_loader = DataLoader(NewsDataset(X_train, y_train, vocab, MAX_SEQ_LEN), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(NewsDataset(X_test, y_test, vocab, MAX_SEQ_LEN), batch_size=BATCH_SIZE)
    
    # Model & Loss
    model = BiLSTMClassifier(len(vocab.word2idx), EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
    
    num_fake = sum(y_train)
    num_real = len(y_train) - num_fake
    pos_weight = torch.tensor([num_real / num_fake], dtype=torch.float).to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\n[Start] Egitim dongusu basladi...")
    best_acc = 0
    for epoch in range(EPOCHS):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        v_loss, v_acc = evaluate(model, test_loader, criterion)
        
        print(f"\n>> EPOCH {epoch+1} SONUCU: Train Acc: %{t_acc*100:.2f} | Test Dogrulugu = %{v_acc*100:.2f}")
        
        if v_acc > best_acc:
            best_acc = v_acc
            save_path = os.path.join("models", "saved", "bilstm_model.pt")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_word2idx': vocab.word2idx,
                'hyperparameters': {'embedding_dim': EMBEDDING_DIM, 'hidden_size': HIDDEN_SIZE, 'num_layers': NUM_LAYERS, 'dropout': DROPOUT, 'max_seq_len': MAX_SEQ_LEN, 'vocab_size': len(vocab.word2idx)}
            }, save_path)
            print(f"   [Save] En iyi model kaydedildi! (Acc: %{v_acc*100:.2f})")

    print("\n" + "="*60)
    print(f" ✅ TAM EGITIM TAMAMLANDI! En iyi dogruluk: %{best_acc*100:.2f}")
    print("="*60)

if __name__ == "__main__":
    main()