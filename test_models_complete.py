#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
اختبار شامل لنماذج كشف الأخبار الزائفة
Testing script for fake news detection models
"""

import sys
import os
import io

# Handle UTF-8 encoding on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import pandas as pd

# إضافة مسار scripts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts')))

from scripts.train_bilstm import BiLSTMClassifier
from scripts.train_bert import BertClassifier
from transformers import BertTokenizer
import re

# ===========================
# تحضير البيانات الاختبارية
# ===========================

test_news = {
    'Real_Turkish': [
        "Türkiye ve Yunanistan arasındaki dostluk ilişkileri derinleşmeye devam ediyor. Ticari işbirliği artırıldı.",
        "Reuters ajansına göre, Ankara'da düzenlenen ekonomi konferansında yeni politikalar açıklandı.",
        "Cumhuriyet gazetesinden alınan haberlere göre, teknik üniversitede yeni bölümler açılacak.",
        "Resmi kaynaklara göre, Marmara denizinde su kalitesi iyileşti.",
        "Ulusal haber ajansının bildirdiğine göre, sağlık alanında yeni teknolojiler kullanılmaya başlanıyor.",
    ],
    'Fake_Turkish': [
        "Şok edici: Dünya hükümetleri uzaylılarla anlaşma yaptı! Son dakika gelişme çok çarpıcı!",
        "İnanılmaz iddia: Ünlü milyarder sosyal ağı kapattı ve gitmiş! Haber duyunca inanamadık!",
        "Mübaşir haberi: Ünlü oyuncu hastaneye kaldırıldı, doktor açıklamada yer almadı!",
        "Bomba haber: Merkez Bankası tüm paralarını çekti, korkunç sır açığa çıktı!",
        "Son dakika: Dünyanın en zengin kişisi tüm servetini bağışladı, nedeni çok enterasan!",
    ],
    'Real_English': [
        "The World Health Organization announced new guidelines for disease prevention.",
        "According to Reuters, the European Union approved a new trade agreement.",
        "Scientists from MIT discovered a new material with potential applications in renewable energy.",
        "The International Court of Justice released its decision on border disputes.",
        "Official statement: The central bank raised interest rates to control inflation.",
    ],
    'Fake_English': [
        "Shocking: Secret government files reveal aliens have been on Earth for centuries!",
        "Breaking: Billionaire tech founder unexpectedly leaves and no one knows where!",
        "Unbelievable: Famous celebrity rushed to hospital, doctors refuse to comment!",
        "Bombshell: New conspiracy discovered - governments hiding truth about our history!",
        "Latest: Richest person in world donates entire fortune, shocking reason revealed!",
    ]
}

# ===========================
# دوال المعالجة
# ===========================

def preprocess_text_for_bilstm(text):
    """معالجة النص لنموذج Bi-LSTM"""
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zçşğüöı\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_with_bilstm(text, model, vocab, device, max_seq_len=256):
    """التنبؤ باستخدام Bi-LSTM"""
    text = preprocess_text_for_bilstm(text)
    tokens = text.split()
    encoded = [vocab.get(word, 1) for word in tokens]

    if len(encoded) < max_seq_len:
        encoded = encoded + [0] * (max_seq_len - len(encoded))
    else:
        encoded = encoded[:max_seq_len]

    tensor = torch.tensor([encoded], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(tensor, apply_sigmoid=True).cpu().item()

    return output

def predict_with_bert(text, model, tokenizer, device, max_length=128):
    """التنبؤ باستخدام BERT"""
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        logit = model(input_ids, attention_mask)
        output = torch.sigmoid(logit).cpu().item()

    return output

# ===========================
# تحميل النماذج
# ===========================

print("=" * 80)
print("جاري تحميل النماذج... Loading models...")
print("=" * 80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# تحميل Bi-LSTM
try:
    bilstm_path = "models/saved/bilstm_model.pt"
    checkpoint = torch.load(bilstm_path, map_location=device, weights_only=False)
    hp = checkpoint['hyperparameters']
    bilstm_model = BiLSTMClassifier(hp['vocab_size'], hp['embedding_dim'], hp['hidden_size'], hp['num_layers'], hp['dropout'])
    bilstm_model.load_state_dict(checkpoint['model_state_dict'])
    bilstm_model.eval()
    bilstm_vocab = checkpoint['vocab_word2idx']
    print("✓ Bi-LSTM Model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load Bi-LSTM: {e}")
    bilstm_model = None

# تحميل BERT
try:
    bert_path = "models/saved/bert_model.pt"
    checkpoint = torch.load(bert_path, map_location=device, weights_only=False)
    model_name = checkpoint.get('model_name', "bert-base-multilingual-cased")
    dropout = checkpoint.get('dropout', 0.3)

    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_model = BertClassifier(model_name, dropout)
    bert_model.load_state_dict(checkpoint['model_state_dict'])
    bert_model.eval()
    print("✓ BERT Model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load BERT: {e}")
    bert_model = None

print()

# ===========================
# إجراء الاختبارات
# ===========================

results = []

for category, texts in test_news.items():
    print("=" * 80)
    print(f"الفئة: {category}")
    print("=" * 80)

    is_fake = 'Fake' in category
    category_results = []

    for i, text in enumerate(texts, 1):
        print(f"\n{i}. {text[:80]}...")

        bilstm_pred = None
        bert_pred = None

        # Bi-LSTM prediction
        if bilstm_model is not None:
            try:
                bilstm_pred = predict_with_bilstm(text, bilstm_model, bilstm_vocab, device)
                bilstm_label = "FAKE" if bilstm_pred > 0.5 else "REAL"
                bilstm_conf = bilstm_pred * 100 if bilstm_pred > 0.5 else (1 - bilstm_pred) * 100
                print(f"   Bi-LSTM: {bilstm_label} (Score: {bilstm_pred:.4f}, Conf: {bilstm_conf:.1f}%)")
            except Exception as e:
                print(f"   Bi-LSTM: Error - {e}")

        # BERT prediction
        if bert_model is not None:
            try:
                bert_pred = predict_with_bert(text, bert_model, tokenizer, device)
                bert_label = "FAKE" if bert_pred > 0.5 else "REAL"
                bert_conf = bert_pred * 100 if bert_pred > 0.5 else (1 - bert_pred) * 100
                print(f"   BERT:    {bert_label} (Score: {bert_pred:.4f}, Conf: {bert_conf:.1f}%)")
            except Exception as e:
                print(f"   BERT: Error - {e}")

        # Store results
        results.append({
            'Category': category,
            'Expected': 'FAKE' if is_fake else 'REAL',
            'Text': text[:60] + "...",
            'BiLSTM_Pred': bilstm_label if bilstm_model else 'N/A',
            'BiLSTM_Score': f"{bilstm_pred:.4f}" if bilstm_pred is not None else 'N/A',
            'BERT_Pred': bert_label if bert_model else 'N/A',
            'BERT_Score': f"{bert_pred:.4f}" if bert_pred is not None else 'N/A',
        })

# ===========================
# ملخص النتائج
# ===========================

print("\n" + "=" * 80)
print("ملخص النتائج Summary of Results")
print("=" * 80)

df = pd.DataFrame(results)
print("\n" + df.to_string(index=False))

# حساب الدقة
if bilstm_model and bert_model:
    print("\n" + "=" * 80)
    print("تقييم الدقة Accuracy Evaluation")
    print("=" * 80)

    bilstm_correct = sum(1 for r in results if r['Expected'] == r['BiLSTM_Pred'])
    bert_correct = sum(1 for r in results if r['Expected'] == r['BERT_Pred'])

    bilstm_acc = (bilstm_correct / len(results)) * 100
    bert_acc = (bert_correct / len(results)) * 100

    print(f"\nBi-LSTM Accuracy: {bilstm_acc:.1f}% ({bilstm_correct}/{len(results)})")
    print(f"BERT Accuracy:    {bert_acc:.1f}% ({bert_correct}/{len(results)})")
    print(f"Average:          {(bilstm_acc + bert_acc) / 2:.1f}%")

print("\n" + "=" * 80)
print("الاختبار انتهى Test completed")
print("=" * 80)
