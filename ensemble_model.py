#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ensemble Model: Combining Bi-LSTM and BERT for Better Predictions
نموذج دمج: جمع Bi-LSTM و BERT للحصول على تنبؤات أفضل
"""

import sys
import os
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import pandas as pd
import re
from enum import Enum

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts')))

from scripts.train_bilstm import BiLSTMClassifier
from scripts.train_bert import BertClassifier
from transformers import BertTokenizer

# ===========================
# Ensemble Strategy Enum
# ===========================

class EnsembleStrategy(Enum):
    AVERAGE = "average"  # (pred1 + pred2) / 2
    WEIGHTED = "weighted"  # BERT_WEIGHT * pred_bert + (1 - BERT_WEIGHT) * pred_bilstm
    MAX = "max"  # max(pred1, pred2) - for high confidence in fake
    MIN = "min"  # min(pred1, pred2) - for conservative fake detection
    VOTING = "voting"  # vote with threshold

# ===========================
# Ensemble Model
# ===========================

class FakeNewsEnsemble:
    """يجمع بين نموذجي Bi-LSTM و BERT للحصول على تنبؤات أفضل"""

    def __init__(self, bilstm_model, bilstm_vocab, bert_model, bert_tokenizer, device='cpu'):
        self.bilstm_model = bilstm_model
        self.bilstm_vocab = bilstm_vocab
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.device = device

        # الأوزان - BERT حصل على دقة أعلى (75%) مقابل Bi-LSTM (55%)
        self.bert_weight = 0.65
        self.bilstm_weight = 0.35

    def preprocess_text_for_bilstm(self, text):
        """معالجة النص لـ Bi-LSTM"""
        if not text:
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zçşğüöı\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def predict_bilstm(self, text, max_seq_len=256):
        """التنبؤ باستخدام Bi-LSTM"""
        if self.bilstm_model is None:
            return None

        try:
            text = self.preprocess_text_for_bilstm(text)
            tokens = text.split()
            encoded = [self.bilstm_vocab.get(word, 1) for word in tokens]

            if len(encoded) < max_seq_len:
                encoded = encoded + [0] * (max_seq_len - len(encoded))
            else:
                encoded = encoded[:max_seq_len]

            tensor = torch.tensor([encoded], dtype=torch.long).to(self.device)

            with torch.no_grad():
                output = self.bilstm_model(tensor, apply_sigmoid=True).cpu().item()

            return output
        except Exception as e:
            print(f"Bi-LSTM Error: {e}")
            return None

    def predict_bert(self, text, max_length=128):
        """التنبؤ باستخدام BERT"""
        if self.bert_model is None:
            return None

        try:
            encoding = self.bert_tokenizer(
                text,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            with torch.no_grad():
                logit = self.bert_model(input_ids, attention_mask)
                output = torch.sigmoid(logit).cpu().item()

            return output
        except Exception as e:
            print(f"BERT Error: {e}")
            return None

    def predict_ensemble(self, text, strategy=EnsembleStrategy.WEIGHTED):
        """
        دمج التنبؤات من النموذجين
        Combine predictions from both models
        """
        bilstm_pred = self.predict_bilstm(text)
        bert_pred = self.predict_bert(text)

        results = {
            'bilstm_score': bilstm_pred,
            'bert_score': bert_pred,
            'bilstm_available': bilstm_pred is not None,
            'bert_available': bert_pred is not None,
        }

        # إذا كان أحد النموذجين فقط متاحاً
        if bilstm_pred is None and bert_pred is not None:
            results['ensemble_score'] = bert_pred
            results['strategy_used'] = 'bert_only'
            return results

        if bert_pred is None and bilstm_pred is not None:
            results['ensemble_score'] = bilstm_pred
            results['strategy_used'] = 'bilstm_only'
            return results

        if bilstm_pred is None and bert_pred is None:
            results['ensemble_score'] = None
            results['strategy_used'] = 'none'
            return results

        # تطبيق استراتيجية الدمج
        if strategy == EnsembleStrategy.AVERAGE:
            ensemble_score = (bilstm_pred + bert_pred) / 2

        elif strategy == EnsembleStrategy.WEIGHTED:
            # استخدام الأوزان بناءً على الدقة
            ensemble_score = self.bert_weight * bert_pred + self.bilstm_weight * bilstm_pred

        elif strategy == EnsembleStrategy.MAX:
            # أقصى قيمة - للحصول على ثقة عالية في أن الخبر زائف
            ensemble_score = max(bilstm_pred, bert_pred)

        elif strategy == EnsembleStrategy.MIN:
            # أقل قيمة - كشف متحفظ
            ensemble_score = min(bilstm_pred, bert_pred)

        elif strategy == EnsembleStrategy.VOTING:
            # تصويت - إذا اتفق النموذجان على نفس النتيجة
            bilstm_label = 1 if bilstm_pred > 0.5 else 0
            bert_label = 1 if bert_pred > 0.5 else 0
            votes = bilstm_label + bert_label
            ensemble_score = (votes / 2)  # 0, 0.5, أو 1

        else:
            ensemble_score = (bilstm_pred + bert_pred) / 2

        results['ensemble_score'] = ensemble_score
        results['strategy_used'] = strategy.value

        return results

    def predict_detailed(self, text, strategy=EnsembleStrategy.WEIGHTED):
        """تنبؤ تفصيلي مع معلومات إضافية"""
        results = self.predict_ensemble(text, strategy)

        ensemble_score = results['ensemble_score']

        if ensemble_score is not None:
            is_fake = ensemble_score > 0.5
            confidence = ensemble_score * 100 if is_fake else (1 - ensemble_score) * 100

            results.update({
                'prediction': 'FAKE' if is_fake else 'REAL',
                'confidence': confidence,
                'ensemble_score_formatted': f"{ensemble_score:.4f}"
            })

        return results


# ===========================
# اختبار Ensemble
# ===========================

if __name__ == "__main__":
    print("=" * 80)
    print("تحميل النماذج لـ Ensemble Loading models for Ensemble...")
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
        print("✓ Bi-LSTM Model loaded")
    except Exception as e:
        print(f"✗ Bi-LSTM Error: {e}")
        bilstm_model = None
        bilstm_vocab = None

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
        print("✓ BERT Model loaded")
    except Exception as e:
        print(f"✗ BERT Error: {e}")
        bert_model = None
        tokenizer = None

    print()

    # إنشء Ensemble
    ensemble = FakeNewsEnsemble(bilstm_model, bilstm_vocab, bert_model, tokenizer, device)

    # أمثلة للاختبار
    test_cases = [
        ("Real", "Reuters ajansına göre, Ankara'da düzenlenen ekonomi konferansında yeni politikalar açıklandı."),
        ("Fake", "Şok edici: Dünya hükümetleri uzaylılarla anlaşma yaptı! Son dakika gelişme çok çarpıcı!"),
        ("Real", "The World Health Organization announced new guidelines for disease prevention."),
        ("Fake", "Shocking: Secret government files reveal aliens have been on Earth for centuries!"),
    ]

    print("=" * 80)
    print("اختبار Ensemble - جميع الاستراتيجيات Testing Ensemble - All Strategies")
    print("=" * 80)

    strategies = [
        EnsembleStrategy.AVERAGE,
        EnsembleStrategy.WEIGHTED,
        EnsembleStrategy.MAX,
        EnsembleStrategy.VOTING,
    ]

    for expected, text in test_cases:
        print(f"\nالمتوقع: {expected}")
        print(f"النص: {text[:70]}...")
        print("-" * 80)

        for strategy in strategies:
            result = ensemble.predict_detailed(text, strategy)

            if result.get('ensemble_score') is not None:
                pred = result['prediction']
                conf = result['confidence']
                print(f"  {strategy.value.upper():12} → {pred:4} (Conf: {conf:.1f}%, Score: {result['ensemble_score_formatted']})")

        print()

    print("=" * 80)
    print("الخلاصة Summary")
    print("=" * 80)
    print("""
الاستراتيجيات المتاحة:
1. AVERAGE      - متوسط النتبؤات من النموذجين (بسيط)
2. WEIGHTED     - متوسط مرجح (BERT: 65%, Bi-LSTM: 35%)
3. MAX          - أقصى درجة (عالي الثقة في كشف الأخبار الزائفة)
4. VOTING       - تصويت (متحفظ - يتطلب اتفاق النموذجين)

التوصية: استخدم WEIGHTED أو VOTING للحصول على أفضل توازن
""")
