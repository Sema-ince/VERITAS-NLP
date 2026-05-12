#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
اختبار شامل: نماذج منفردة + Ensemble
Comprehensive Test: Individual Models + Ensemble
"""

import sys
import os
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import pandas as pd
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts')))

from scripts.train_bilstm import BiLSTMClassifier
from scripts.train_bert import BertClassifier
from transformers import BertTokenizer

# ===========================
# Test Data
# ===========================

test_data = {
    'Turkish_Real': [
        "Reuters ajansına göre, Ankara'da düzenlenen ekonomi konferansında yeni politikalar açıklandı.",
        "Cumhuriyet gazetesinden alınan haberlere göre, teknik üniversitede yeni bölümler açılacak.",
        "Resmi kaynaklara göre, Marmara denizinde su kalitesi iyileşti.",
    ],
    'Turkish_Fake': [
        "Şok edici: Dünya hükümetleri uzaylılarla anlaşma yaptı! Son dakika gelişme çok çarpıcı!",
        "İnanılmaz iddia: Ünlü milyarder sosyal ağı kapattı ve gitmiş!",
        "Bomba haber: Merkez Bankası tüm paralarını çekti, korkunç sır açığa çıktı!",
    ],
    'English_Real': [
        "According to Reuters, the European Union approved a new trade agreement.",
        "The International Court of Justice released its decision on border disputes.",
        "Official statement: The central bank raised interest rates to control inflation.",
    ],
    'English_Fake': [
        "Shocking: Secret government files reveal aliens have been on Earth for centuries!",
        "Breaking: Billionaire tech founder unexpectedly leaves and no one knows where!",
        "Bombshell: New conspiracy discovered - governments hiding truth about our history!",
    ]
}

# ===========================
# Helper Functions
# ===========================

def preprocess_text_for_bilstm(text):
    """معالجة النص لـ Bi-LSTM"""
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zçşğüöı\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ===========================
# Load Models
# ===========================

print("\n" + "="*80)
print("تحميل النماذج Loading Models...")
print("="*80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Bi-LSTM
bilstm_model = None
bilstm_vocab = None
bilstm_improved = False

try:
    # محاولة تحميل النسخة المحسّنة أولاً
    bilstm_path = "models/saved/bilstm_model_improved.pt"
    if os.path.exists(bilstm_path):
        checkpoint = torch.load(bilstm_path, map_location=device, weights_only=False)
        hp = checkpoint['hyperparameters']
        from scripts.train_bilstm_improved import ImprovedBiLSTMClassifier
        bilstm_model = ImprovedBiLSTMClassifier(hp['vocab_size'], hp['embedding_dim'], hp['hidden_size'], hp['num_layers'], hp['dropout'])
        bilstm_improved = True
        print("✓ Bi-LSTM Improved Model loaded")
    else:
        # تحميل النسخة القديمة
        bilstm_path = "models/saved/bilstm_model.pt"
        checkpoint = torch.load(bilstm_path, map_location=device, weights_only=False)
        hp = checkpoint['hyperparameters']
        bilstm_model = BiLSTMClassifier(hp['vocab_size'], hp['embedding_dim'], hp['hidden_size'], hp['num_layers'], hp['dropout'])
        print("✓ Bi-LSTM Original Model loaded")

    bilstm_model.load_state_dict(checkpoint['model_state_dict'])
    bilstm_model.eval()
    bilstm_vocab = checkpoint['vocab_word2idx']
except Exception as e:
    print(f"✗ Bi-LSTM Error: {e}")

# BERT
try:
    bert_path = "models/saved/bert_model.pt"
    checkpoint = torch.load(bert_path, map_location=device, weights_only=False)
    model_name = checkpoint.get('model_name', "bert-base-multilingual-cased")
    dropout = checkpoint.get('dropout', 0.3)

    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_model = BertClassifier(model_name, dropout)
    bert_model.load_state_dict(checkpoint['model_state_dict'])
    bert_model.eval()
    print("✓ BERT Model loaded")
except Exception as e:
    print(f"✗ BERT Error: {e}")
    bert_model = None
    tokenizer = None

# ===========================
# Prediction Functions
# ===========================

def predict_bilstm(text):
    """التنبؤ باستخدام Bi-LSTM"""
    if bilstm_model is None:
        return None

    try:
        text = preprocess_text_for_bilstm(text)
        tokens = text.split()
        encoded = [bilstm_vocab.get(word, 1) for word in tokens]
        max_seq_len = 256 if not bilstm_improved else 128

        if len(encoded) < max_seq_len:
            encoded = encoded + [0] * (max_seq_len - len(encoded))
        else:
            encoded = encoded[:max_seq_len]

        tensor = torch.tensor([encoded], dtype=torch.long).to(device)

        with torch.no_grad():
            output = bilstm_model(tensor, apply_sigmoid=True).cpu().item()

        return output
    except Exception as e:
        return None

def predict_bert(text):
    """التنبؤ باستخدام BERT"""
    if bert_model is None:
        return None

    try:
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            logit = bert_model(input_ids, attention_mask)
            output = torch.sigmoid(logit).cpu().item()

        return output
    except Exception as e:
        return None

def predict_ensemble_weighted(bilstm_score, bert_score):
    """دمج النتبؤات بطريقة الأوزان المرجحة"""
    if bilstm_score is None and bert_score is None:
        return None
    if bilstm_score is None:
        return bert_score
    if bert_score is None:
        return bilstm_score

    bert_weight = 0.65
    bilstm_weight = 0.35
    return bert_weight * bert_score + bilstm_weight * bilstm_score

# ===========================
# Run Tests
# ===========================

results = []

print("\n" + "="*80)
print("الاختبار Testing")
print("="*80)

for category, texts in test_data.items():
    is_fake = 'Fake' in category
    expected = 'FAKE' if is_fake else 'REAL'

    print(f"\n{category} (Expected: {expected})")
    print("-" * 80)

    for i, text in enumerate(texts, 1):
        bilstm_score = predict_bilstm(text)
        bert_score = predict_bert(text)
        ensemble_score = predict_ensemble_weighted(bilstm_score, bert_score)

        print(f"\n  {i}. {text[:60]}...")

        if bilstm_score is not None:
            bilstm_pred = "FAKE" if bilstm_score > 0.5 else "REAL"
            bilstm_conf = bilstm_score * 100 if bilstm_score > 0.5 else (1 - bilstm_score) * 100
            print(f"     Bi-LSTM: {bilstm_pred:4} ({bilstm_conf:5.1f}%) - Score: {bilstm_score:.4f}")

        if bert_score is not None:
            bert_pred = "FAKE" if bert_score > 0.5 else "REAL"
            bert_conf = bert_score * 100 if bert_score > 0.5 else (1 - bert_score) * 100
            print(f"     BERT:    {bert_pred:4} ({bert_conf:5.1f}%) - Score: {bert_score:.4f}")

        if ensemble_score is not None:
            ensemble_pred = "FAKE" if ensemble_score > 0.5 else "REAL"
            ensemble_conf = ensemble_score * 100 if ensemble_score > 0.5 else (1 - ensemble_score) * 100
            print(f"     Ensemble:{ensemble_pred:4} ({ensemble_conf:5.1f}%) - Score: {ensemble_score:.4f} ✓")

        results.append({
            'Category': category,
            'Expected': expected,
            'BiLSTM': 'FAKE' if bilstm_score and bilstm_score > 0.5 else 'REAL' if bilstm_score else 'N/A',
            'BERT': 'FAKE' if bert_score and bert_score > 0.5 else 'REAL' if bert_score else 'N/A',
            'Ensemble': 'FAKE' if ensemble_score and ensemble_score > 0.5 else 'REAL' if ensemble_score else 'N/A',
        })

# ===========================
# Summary
# ===========================

print("\n" + "="*80)
print("ملخص النتائج Summary")
print("="*80)

df = pd.DataFrame(results)
print("\n" + df.to_string(index=False))

# Accuracy
if len(results) > 0:
    print("\n" + "="*80)
    print("الدقة Accuracy")
    print("="*80)

    bilstm_acc = sum(1 for r in results if r['Expected'] == r['BiLSTM']) / len(results) * 100 if 'N/A' not in [r['BiLSTM'] for r in results] else 0
    bert_acc = sum(1 for r in results if r['Expected'] == r['BERT']) / len(results) * 100 if 'N/A' not in [r['BERT'] for r in results] else 0
    ensemble_acc = sum(1 for r in results if r['Expected'] == r['Ensemble']) / len(results) * 100 if 'N/A' not in [r['Ensemble'] for r in results] else 0

    print(f"\nBi-LSTM Accuracy:  {bilstm_acc:.1f}%")
    print(f"BERT Accuracy:     {bert_acc:.1f}%")
    print(f"Ensemble Accuracy: {ensemble_acc:.1f}%")

    if ensemble_acc > bert_acc and ensemble_acc > bilstm_acc:
        print("\n✓ Ensemble يعطي أفضل أداء!")
    elif bert_acc >= ensemble_acc and bert_acc >= bilstm_acc:
        print("\n✓ BERT يعطي أفضل أداء!")

print("\n" + "="*80)

model_status = "محسّن (Improved)" if bilstm_improved else "أصلي (Original)"
print(f"\nBi-LSTM Status: {model_status}")
print("="*80)
