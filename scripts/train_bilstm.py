"""
VERITAS-NLP: Bi-LSTM Derin Öğrenme Modeli Eğitimi
==================================================
Bu dosya, Ahmet tarafından hazırlanan Baseline modellerinin (Logistic Regression & Random Forest)
üzerine çıkmayı hedefleyen derin öğrenme modelidir.

Kullanılan Veri: Sinan'ın temizlediği WELFake_cleaned.csv
Mimari: Embedding -> Bi-LSTM -> Dense (Tam Bağlantılı Katman)
Framework: PyTorch
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

# =============================================================
# 1. AYARLAR (Hyperparameters) - Literatür Taramasından Alındı
# =============================================================
EMBEDDING_DIM = 128       # Her kelimenin vektör boyutu
HIDDEN_SIZE = 128         # Bi-LSTM'in gizli katman boyutu (Literatür: 128 veya 256)
NUM_LAYERS = 2            # Bi-LSTM katman sayısı (Literatür: 1 veya 2)
DROPOUT = 0.3             # Overfitting'i önlemek için (Literatür: 0.3)
MAX_VOCAB_SIZE = 20000    # En sık kullanılan 20k kelime
MAX_SEQ_LEN = 256         # Bir haberin maksimum kelime uzunluğu (Literatür: 128 veya 256)

BATCH_SIZE = 32           # (Literatür: 16 veya 32)
EPOCHS = 5                # (Literatür: 3-5)
LEARNING_RATE = 1e-3      # Bi-LSTM için standart LR (BERT'ten farklı, orada 2e-5)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================
# 2. KELİME SÖZLÜĞÜ OLUŞTURMA (Vocabulary)
# =============================================================
class Vocabulary:
    """
    Metindeki kelimeleri sayılara çevirir.
    Örnek: "haber sahte" -> [45, 312]
    """
    def __init__(self, max_size):
        self.max_size = max_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}  # PAD=boşluk doldurma, UNK=bilinmeyen kelime
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
    
    def build(self, texts):
        """Tüm metinlerden en sık kullanılan kelimelerin sözlüğünü oluşturur."""
        word_counts = Counter()
        for text in texts:
            word_counts.update(text.split())
        
        # En sık kullanılan kelimeleri al (max_size - 2 çünkü PAD ve UNK zaten var)
        most_common = word_counts.most_common(self.max_size - 2)
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        print(f"📚 Sözlük oluşturuldu: {len(self.word2idx)} kelime")
    
    def encode(self, text):
        """Bir metni sayı dizisine çevirir."""
        tokens = text.split()
        encoded = [self.word2idx.get(word, 1) for word in tokens]  # Bilinmeyen kelime = 1 (UNK)
        return encoded

# =============================================================
# 3. VERİ SETİ SINIFI (PyTorch Dataset)
# =============================================================
class NewsDataset(Dataset):
    """
    PyTorch'un veriyi parça parça (batch) yüklemesi için özel Dataset sınıfı.
    """
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Metni sayılara çevir
        encoded = self.vocab.encode(self.texts[idx])
        
        # Padding: Kısa metinleri sıfırla doldur, uzun metinleri kes
        if len(encoded) < self.max_len:
            encoded = encoded + [0] * (self.max_len - len(encoded))
        else:
            encoded = encoded[:self.max_len]
        
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float)
        )

# =============================================================
# 4. Bi-LSTM MODEL MİMARİSİ
# =============================================================
class BiLSTMClassifier(nn.Module):
    """
    Sahte Haber Tespiti için Bi-LSTM Modeli.
    
    Mimari (3 Aşamalı Fabrika):
    ─────────────────────────────────────────
    Aşama 1 - Embedding:   Kelimeleri anlamlı vektörlere çevirir
    Aşama 2 - Bi-LSTM:     Metni hem ileri hem geri okuyarak bağlamı anlar
    Aşama 3 - Dense (FC):  Son karar: "Gerçek mi, Sahte mi?"
    ─────────────────────────────────────────
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout):
        super(BiLSTMClassifier, self).__init__()
        
        # Aşama 1: Embedding Katmanı
        # Her kelimeyi (sayıyı) anlamlı bir vektöre dönüştürür
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0  # PAD tokenı için sıfır vektör
        )
        
        # Aşama 2: Bi-LSTM Katmanı
        # Metni iki yönde okur (ileri + geri) ve bağlamı öğrenir
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,   # ÇİFT YÖNLÜ (Bi-directional) aktif!
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Aşama 3: Dense (Tam Bağlantılı) Katmanlar
        # Bi-LSTM çıktısını alıp son kararı verir
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)  # *2 çünkü bidirectional (ileri + geri)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Adım 1: Kelimeleri vektörlere çevir
        embedded = self.embedding(x)            # [batch, seq_len, embedding_dim]
        
        # Adım 2: Bi-LSTM'den geçir
        lstm_out, (hidden, _) = self.lstm(embedded)  # lstm_out: [batch, seq_len, hidden*2]
        
        # Son gizli durumları birleştir (ileri yönün son + geri yönün son)
        hidden_forward = hidden[-2]   # İleri yönün son katmanı
        hidden_backward = hidden[-1]  # Geri yönün son katmanı
        combined = torch.cat((hidden_forward, hidden_backward), dim=1)  # [batch, hidden*2]
        
        # Adım 3: Dropout uygula ve son kararı ver
        combined = self.dropout(combined)
        output = self.fc(combined)
        output = self.sigmoid(output)
        return output.squeeze(1)

# =============================================================
# 5. EĞİTİM FONKSİYONU
# =============================================================
def train_one_epoch(model, dataloader, criterion, optimizer):
    """Modeli bir epoch boyunca eğitir."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_texts, batch_labels in dataloader:
        batch_texts = batch_texts.to(DEVICE)
        batch_labels = batch_labels.to(DEVICE)
        
        # İleri hesaplama
        predictions = model(batch_texts)
        loss = criterion(predictions, batch_labels)
        
        # Geri yayılım (Backpropagation)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend((predictions > 0.5).cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy

# =============================================================
# 6. TEST/DEĞERLENDİRME FONKSİYONU
# =============================================================
def evaluate(model, dataloader, criterion):
    """Modeli test verileri üzerinde değerlendirir."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_texts, batch_labels in dataloader:
            batch_texts = batch_texts.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            
            predictions = model(batch_texts)
            loss = criterion(predictions, batch_labels)
            
            total_loss += loss.item()
            all_preds.extend((predictions > 0.5).cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy, all_preds, all_labels

# =============================================================
# 7. ANA FONKSİYON
# =============================================================
def main():
    print("=" * 60)
    print(" VERITAS-NLP: Bi-LSTM Derin Öğrenme Modeli Eğitimi")
    print("=" * 60)
    print(f"\n🖥️  Kullanılan cihaz: {DEVICE}")
    
    # --- Veri Yükleme ---
    file_path = os.path.join("data", "processed", "WELFake_cleaned.csv")
    
    if not os.path.exists(file_path):
        print(f"\n❌ HATA: Temizlenmiş veri seti bulunamadı -> {file_path}")
        print("Lütfen önce: python scripts/preprocess_welfake.py komutunu çalıştırın.")
        return
    
    print(f"\n📂 Veri yükleniyor: {file_path}")
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['content', 'label'])
    print(f"✅ Toplam {len(df):,} haber yüklendi.")
    
    # --- Veriyi Bölme ---
    print("\n🔀 Veri seti bölünüyor (%80 Eğitim, %20 Test)...")
    texts = df['content'].tolist()
    labels = df['label'].tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=0.2, random_state=42
    )
    
    # --- Sözlük Oluşturma ---
    print("\n📖 Kelime sözlüğü oluşturuluyor...")
    vocab = Vocabulary(MAX_VOCAB_SIZE)
    vocab.build(X_train)
    
    # --- DataLoader Oluşturma ---
    train_dataset = NewsDataset(X_train, y_train, vocab, MAX_SEQ_LEN)
    test_dataset = NewsDataset(X_test, y_test, vocab, MAX_SEQ_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"📦 Eğitim batch sayısı: {len(train_loader)}, Test batch sayısı: {len(test_loader)}")
    
    # --- Model Oluşturma ---
    print("\n🏗️  Bi-LSTM modeli oluşturuluyor...")
    model = BiLSTMClassifier(
        vocab_size=len(vocab.word2idx),
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)
    
    criterion = nn.BCELoss()  # Binary Cross-Entropy (Literatür: Cross-Entropy)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Toplam parametre sayısı: {total_params:,}")
    
    # --- Eğitim Döngüsü ---
    print("\n" + "=" * 60)
    print(" EĞİTİM BAŞLIYOR")
    print("=" * 60)
    
    best_accuracy = 0.0
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion)
        
        print(f"\n📈 Epoch [{epoch+1}/{EPOCHS}]")
        print(f"   Eğitim   -> Kayıp: {train_loss:.4f} | Doğruluk: %{train_acc*100:.2f}")
        print(f"   Test     -> Kayıp: {test_loss:.4f} | Doğruluk: %{test_acc*100:.2f}")
        
        # En iyi modeli kaydet
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            save_dir = os.path.join("models", "saved")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "bilstm_model.pt")
            
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
            print(f"   💾 En iyi model kaydedildi! (Doğruluk: %{best_accuracy*100:.2f})")
    
    # --- Son Değerlendirme ---
    print("\n" + "=" * 60)
    print(" SONUÇLAR")
    print("=" * 60)
    
    _, _, final_preds, final_labels = evaluate(model, test_loader, criterion)
    print(f"\n🏆 En iyi Test Doğruluğu: %{best_accuracy*100:.2f}")
    print("\n📋 Sınıflandırma Raporu:")
    print(classification_report(
        final_labels, final_preds,
        target_names=["Real (0)", "Fake (1)"],
        zero_division=0
    ))
    print(f"💾 Model kaydedildi: models/saved/bilstm_model.pt")

if __name__ == "__main__":
    main()
