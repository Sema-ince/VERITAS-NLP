"""
VERITAS-NLP: BERT Derin Öğrenme Modeli Eğitimi
===============================================
Bu dosya, Literatür Taramasında belirlenen BERT modelini eğitmek için hazırlanmıştır.

Kullanılan Veri: Sinan'ın temizlediği WELFake_cleaned.csv
Model: bert-base-turkish-cased (Literatür Taramasından)
Framework: PyTorch + HuggingFace Transformers
Optimizer: AdamW (Literatür: AdamW)
Learning Rate: 2e-5 (Literatür: 2e-5 civarı)
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# =============================================================
# 1. AYARLAR (Hyperparameters) - Literatür Taramasından Alındı
# =============================================================
MODEL_NAME = "dbmdz/bert-base-turkish-cased"  # Literatür: bert-base-turkish-cased
MAX_LENGTH = 128          # Token uzunluğu (Literatür: 128 veya 256)
DROPOUT = 0.3             # (Literatür: 0.3)

BATCH_SIZE = 16           # (Literatür: 16 veya 32) — BERT için 16 RAM açısından daha güvenli
EPOCHS = 3                # (Literatür: 3-5) — BERT için 3 genelde yeterli
LEARNING_RATE = 2e-5      # (Literatür: 2e-5 civarı)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================
# 2. VERİ SETİ SINIFI (BERT için özel)
# =============================================================
class BertNewsDataset(Dataset):
    """
    BERT modeli için özel Dataset sınıfı.
    BERT'in kendi tokenizer'ı ile metinleri işler.
    """
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # BERT Tokenizer: Metni BERT'in anlayacağı formata çevirir
        # - input_ids: Kelimelerin sayısal karşılıkları
        # - attention_mask: Gerçek kelime mi yoksa padding mi? (1=gerçek, 0=padding)
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,       # [CLS] ve [SEP] tokenlarını ekle
            max_length=self.max_length,
            padding='max_length',           # Kısa metinleri doldur
            truncation=True,                # Uzun metinleri kes
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }

# =============================================================
# 3. BERT SINIFLANDIRMA MODELİ
# =============================================================
class BertClassifier(nn.Module):
    """
    Sahte Haber Tespiti için BERT Modeli.
    
    Mimari:
    ─────────────────────────────────────────
    Aşama 1 - BERT:    Önceden eğitilmiş dil modelini kullanarak
                        metinden derin anlam çıkarır (Fine-tuning)
    Aşama 2 - Dropout: Ezberlemeyi (overfitting) önler
    Aşama 3 - Dense:   Son karar: "Gerçek mi, Sahte mi?"
    ─────────────────────────────────────────
    """
    def __init__(self, model_name, dropout):
        super(BertClassifier, self).__init__()
        
        # Önceden eğitilmiş BERT modelini yükle
        self.bert = BertModel.from_pretrained(model_name)
        
        # Sınıflandırma katmanları
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)  # 768 -> 1
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        # BERT'ten çıktı al
        # [CLS] tokenının çıktısı tüm metnin özetini taşır
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # [CLS] tokenının vektörü (pooled_output)
        pooled_output = bert_output.pooler_output  # [batch, 768]
        
        # Dropout + Sınıflandırma
        output = self.dropout(pooled_output)
        output = self.fc(output)
        output = self.sigmoid(output)
        return output.squeeze(1)

# =============================================================
# 4. EĞİTİM FONKSİYONU
# =============================================================
def train_one_epoch(model, dataloader, criterion, optimizer, scheduler):
    """BERT modelini bir epoch boyunca eğitir."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['label'].to(DEVICE)
        
        # İleri hesaplama
        predictions = model(input_ids, attention_mask)
        loss = criterion(predictions, labels)
        
        # Geri yayılım
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient patlamasını önle
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        all_preds.extend((predictions > 0.5).cpu().detach().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Her 100 batch'te bir ilerleme göster
        if (batch_idx + 1) % 100 == 0:
            print(f"      Batch [{batch_idx+1}/{len(dataloader)}] tamamlandı...")
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy

# =============================================================
# 5. TEST/DEĞERLENDİRME FONKSİYONU
# =============================================================
def evaluate(model, dataloader, criterion):
    """BERT modelini test verileri üzerinde değerlendirir."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, labels)
            
            total_loss += loss.item()
            all_preds.extend((predictions > 0.5).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy, all_preds, all_labels

# =============================================================
# 6. ANA FONKSİYON
# =============================================================
def main():
    print("=" * 60)
    print(" VERITAS-NLP: BERT Derin Öğrenme Modeli Eğitimi")
    print("=" * 60)
    print(f"\n🖥️  Kullanılan cihaz: {DEVICE}")
    print(f"📦 Kullanılan model: {MODEL_NAME}")
    
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
    
    # --- BERT Tokenizer Yükleme ---
    print(f"\n🔤 BERT Tokenizer yükleniyor ({MODEL_NAME})...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    print("✅ Tokenizer hazır.")
    
    # --- Veriyi Bölme ---
    print("\n🔀 Veri seti bölünüyor (%80 Eğitim, %20 Test)...")
    texts = df['content'].tolist()
    labels = df['label'].tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=0.2, random_state=42
    )
    
    # --- DataLoader Oluşturma ---
    train_dataset = BertNewsDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    test_dataset = BertNewsDataset(X_test, y_test, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"📦 Eğitim batch sayısı: {len(train_loader)}, Test batch sayısı: {len(test_loader)}")
    
    # --- Model Oluşturma ---
    print(f"\n🏗️  BERT modeli oluşturuluyor ({MODEL_NAME})...")
    model = BertClassifier(MODEL_NAME, DROPOUT).to(DEVICE)
    
    # Literatürden alınan ayarlar
    criterion = nn.BCELoss()                              # Cross-Entropy
    optimizer = torch.optim.AdamW(                        # Optimizer: AdamW
        model.parameters(), lr=LEARNING_RATE
    )
    
    # Öğrenme oranı zamanlaması (Warm-up)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Toplam parametre: {total_params:,}")
    print(f"📊 Eğitilebilir parametre: {trainable_params:,}")
    
    # --- Eğitim Döngüsü ---
    print("\n" + "=" * 60)
    print(" EĞİTİM BAŞLIYOR")
    print("=" * 60)
    
    best_accuracy = 0.0
    
    for epoch in range(EPOCHS):
        print(f"\n📈 Epoch [{epoch+1}/{EPOCHS}] başlıyor...")
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler
        )
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion)
        
        print(f"   Eğitim   -> Kayıp: {train_loss:.4f} | Doğruluk: %{train_acc*100:.2f}")
        print(f"   Test     -> Kayıp: {test_loss:.4f} | Doğruluk: %{test_acc*100:.2f}")
        
        # En iyi modeli kaydet
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            save_dir = os.path.join("models", "saved")
            os.makedirs(save_dir, exist_ok=True)
            
            # Model ağırlıklarını kaydet
            model_save_path = os.path.join(save_dir, "bert_model.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_name': MODEL_NAME,
                'max_length': MAX_LENGTH,
                'dropout': DROPOUT,
            }, model_save_path)
            
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
    print(f"💾 Model kaydedildi: models/saved/bert_model.pt")

if __name__ == "__main__":
    main()
