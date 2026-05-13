"""
VERITAS-NLP: اختبار شامل للنماذج
=================================
يختبر نموذجي Bi-LSTM و BERT على أخبار حقيقية وكاذبة
باللغتين الإنجليزية والتركية.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import torch
import json

# ============================================================
# تحميل النماذج
# ============================================================
from scripts.train_bilstm import BiLSTMClassifier

def load_bilstm():
    path = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved', 'bilstm_model.pt')
    if not os.path.exists(path):
        print("ERROR: bilstm_model.pt not found")
        return None, None
    checkpoint = torch.load(path, map_location='cpu')
    hp = checkpoint['hyperparameters']
    model = BiLSTMClassifier(hp['vocab_size'], hp['embedding_dim'], hp['hidden_size'], hp['num_layers'], hp['dropout'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['vocab_word2idx']

def load_bert():
    from scripts.train_bert import BertClassifier
    from transformers import BertTokenizer
    path = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved', 'bert_model.pt')
    if not os.path.exists(path):
        print("ERROR: bert_model.pt not found")
        return None, None
    checkpoint = torch.load(path, map_location='cpu')
    model_name = checkpoint.get('model_name', "bert-base-multilingual-cased")
    dropout = checkpoint.get('dropout', 0.3)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertClassifier(model_name, dropout)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, tokenizer

# ============================================================
# دوال التنبؤ (من app.py)
# ============================================================
import re

def preprocess_text_for_bilstm(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zçşğüöı\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_with_bilstm(text, model, vocab, max_seq_len=256):
    device = next(model.parameters()).device
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

def predict_with_bert(text, model, tokenizer, max_length=128):
    device = next(model.parameters()).device
    encoding = tokenizer(
        text, add_special_tokens=True, max_length=max_length,
        padding='max_length', truncation=True,
        return_attention_mask=True, return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        output = model(input_ids, attention_mask).cpu().item()
    return output

def detect_language(text):
    turkish_chars = set('çşğüöıÇŞĞÜÖİ')
    text_chars = set(text)
    if text_chars & turkish_chars:
        return 'tr'
    turkish_words = {'bir', 've', 'ile', 'olan', 'için', 'bu', 'da', 'de',
                     'den', 'dan', 'olarak', 'ise', 'gibi', 'daha', 'sonra'}
    words = set(text.lower().split())
    if len(words & turkish_words) >= 2:
        return 'tr'
    return 'en'

# ============================================================
# مجموعات الاختبار
# ============================================================

# --- أخبار إنجليزية حقيقية (مصادر موثوقة) ---
REAL_NEWS_EN = [
    {
        "text": "The United Nations General Assembly voted on Tuesday to adopt a resolution calling for a humanitarian ceasefire in Gaza. The resolution was supported by 153 countries, with 10 voting against and 23 abstaining. Secretary-General Antonio Guterres welcomed the outcome.",
        "label": "REAL",
        "lang": "en",
        "source": "UN/Reuters style"
    },
    {
        "text": "NASA's Artemis II mission is scheduled to launch in September 2025, sending four astronauts around the Moon for the first time in over 50 years. The crew includes Commander Reid Wiseman, pilot Victor Glover, and mission specialists Christina Koch and Jeremy Hansen. The mission will last approximately 10 days.",
        "label": "REAL",
        "lang": "en",
        "source": "NASA official style"
    },
    {
        "text": "The Federal Reserve announced Wednesday that it would hold interest rates steady at 5.25 to 5.5 percent, citing continued progress on inflation but noting that economic uncertainties remain. Chair Jerome Powell said the central bank would continue to monitor incoming data before making any changes to monetary policy.",
        "label": "REAL",
        "lang": "en",
        "source": "AP/Financial news style"
    },
    {
        "text": "Researchers at MIT published a new study in the journal Nature showing that a novel mRNA-based therapy can significantly reduce tumor growth in mice with pancreatic cancer. The study involved 200 laboratory mice over a period of 18 months and showed a 70 percent reduction in tumor size compared to control groups.",
        "label": "REAL",
        "lang": "en",
        "source": "Academic/Science news style"
    },
]

# --- أخبار إنجليزية كاذبة (نمط تضليل واضح) ---
FAKE_NEWS_EN = [
    {
        "text": "BREAKING: Scientists discover that 5G towers are secretly spreading a new virus that turns people into zombies! The government is hiding this from the public. Share this before they delete it! Exposed secret documents reveal the shocking truth that mainstream media won't tell you.",
        "label": "FAKE",
        "lang": "en",
        "source": "Conspiracy style"
    },
    {
        "text": "SHOCKING: Celebrity doctor reveals miracle cure that Big Pharma doesn't want you to know about! This one simple trick can cure cancer, diabetes, and heart disease overnight. Exposed: the secret they are hiding from you. Doctors hate this discovery!",
        "label": "FAKE",
        "lang": "en",
        "source": "Clickbait/Health misinformation"
    },
    {
        "text": "URGENT: The moon landing was faked by NASA in a Hollywood studio! Former astronaut confesses everything on deathbed. Secret leaked footage proves the whole thing was a hoax. The mainstream media is covering this up. Wake up people!",
        "label": "FAKE",
        "lang": "en",
        "source": "Conspiracy theory"
    },
    {
        "text": "EXCLUSIVE: Underground bunkers discovered beneath the White House containing alien technology! Government whistleblower exposes massive cover-up spanning decades. The elite don't want you to know the truth. This bombshell report reveals everything they've been hiding.",
        "label": "FAKE",
        "lang": "en",
        "source": "Fake sensationalism"
    },
]

# --- أخبار تركية حقيقية ---
REAL_NEWS_TR = [
    {
        "text": "Türkiye İstatistik Kurumu (TÜİK), 2024 yılı enflasyon verilerini açıkladı. Buna göre yıllık tüketici fiyat endeksi yüzde 44,38 olarak gerçekleşti. Gıda ve alkolsüz içecekler grubunda ise yıllık artış yüzde 49,12 oldu. TÜİK Başkanı konuyla ilgili basın toplantısı düzenledi.",
        "label": "REAL",
        "lang": "tr",
        "source": "TÜİK resmi stil"
    },
    {
        "text": "İstanbul Büyükşehir Belediyesi, şehir içi ulaşımda yeni metro hattının açılışını gerçekleştirdi. Başakşehir-Kayaşehir metro hattı günlük 500 bin yolcuya hizmet verecek. Belediye Başkanı açılış töreninde konuştu ve projenin detaylarını paylaştı.",
        "label": "REAL",
        "lang": "tr",
        "source": "Belediye haberi"
    },
    {
        "text": "Cumhurbaşkanlığı Kabinesi toplantısı sonrasında alınan kararlar açıklandı. Toplantıda eğitim reformu, sağlık yatırımları ve tarım destekleri ele alındı. Cumhurbaşkanı yardımcısı düzenlediği basın toplantısında kararları kamuoyuyla paylaştı.",
        "label": "REAL",
        "lang": "tr",
        "source": "Resmi haber"
    },
]

# --- أخبار تركية كاذبة ---
FAKE_NEWS_TR = [
    {
        "text": "ŞOK İDDİA: Hükümet gizli belgeleri sızdırdı! Vatandaşlara gizlice çip takıldığı ortaya çıktı. Bu şok edici gerçeği kimse size söylemeyecek. Medya bu haberi sansürlüyor! Paylaşmadan önce silinecek!",
        "label": "FAKE",
        "lang": "tr",
        "source": "Komplo teorisi"
    },
    {
        "text": "FLAŞ: Dünyaca ünlü profesör açıkladı, bu bitki çayı kanserı tamamen yok ediyor! Doktorlar bu gerçeği sizden saklıyor çünkü ilaç şirketleri para kazanmak istiyor. Bu mucize tedaviyi herkes bilmeli! Gizli belgeler sızdırıldı!",
        "label": "FAKE",
        "lang": "tr",
        "source": "Sağlık dezenformasyonu"
    },
    {
        "text": "İNANILMAZ: Uzaylılar Türkiye'ye indi ve hükümetle gizli anlaşma yaptı! Eski askeri yetkili her şeyi itiraf etti. Bu bomba haber ana akım medyada asla yayınlanmayacak. Gerçekleri öğrenmek için paylaşın!",
        "label": "FAKE",
        "lang": "tr",
        "source": "Sahte sansasyon"
    },
]

# ============================================================
# الاختبار الرئيسي
# ============================================================
def main():
    print("=" * 70)
    print("  VERITAS-NLP: COMPREHENSIVE MODEL TESTING")
    print("=" * 70)

    # تحميل النماذج
    print("\n[1/3] Loading Bi-LSTM model...")
    bilstm_model, bilstm_vocab = load_bilstm()
    bilstm_ok = bilstm_model is not None
    print(f"  -> Bi-LSTM: {'LOADED' if bilstm_ok else 'FAILED'}")

    print("\n[2/3] Loading BERT model...")
    bert_model, bert_tokenizer = load_bert()
    bert_ok = bert_model is not None
    print(f"  -> BERT: {'LOADED' if bert_ok else 'FAILED'}")

    if not bilstm_ok and not bert_ok:
        print("\nERROR: No models available. Aborting.")
        return

    # جمع كل الأخبار
    all_tests = []
    all_tests += [(n, "EN_REAL") for n in REAL_NEWS_EN]
    all_tests += [(n, "EN_FAKE") for n in FAKE_NEWS_EN]
    all_tests += [(n, "TR_REAL") for n in REAL_NEWS_TR]
    all_tests += [(n, "TR_FAKE") for n in FAKE_NEWS_TR]

    results = []

    print(f"\n[3/3] Running predictions on {len(all_tests)} test cases...\n")
    print("-" * 70)

    for i, (news, category) in enumerate(all_tests, 1):
        text = news["text"]
        expected = news["label"]
        lang = news["lang"]
        source = news["source"]
        detected_lang = detect_language(text)

        bilstm_score = None
        bert_score = None

        if bilstm_ok:
            bilstm_score = predict_with_bilstm(text, bilstm_model, bilstm_vocab)
        if bert_ok:
            bert_score = predict_with_bert(text, bert_model, bert_tokenizer)

        # تحديد النتيجة
        bilstm_pred = None
        bert_pred = None
        if bilstm_score is not None:
            bilstm_pred = "FAKE" if bilstm_score > 0.5 else "REAL"
        if bert_score is not None:
            bert_pred = "FAKE" if bert_score > 0.5 else "REAL"

        bilstm_correct = (bilstm_pred == expected) if bilstm_pred else None
        bert_correct = (bert_pred == expected) if bert_pred else None

        result = {
            "id": i,
            "category": category,
            "expected": expected,
            "lang": lang,
            "detected_lang": detected_lang,
            "source": source,
            "bilstm_score": round(bilstm_score, 4) if bilstm_score is not None else None,
            "bilstm_pred": bilstm_pred,
            "bilstm_correct": bilstm_correct,
            "bert_score": round(bert_score, 4) if bert_score is not None else None,
            "bert_pred": bert_pred,
            "bert_correct": bert_correct,
            "text_preview": text[:80] + "..."
        }
        results.append(result)

        # طباعة النتيجة
        bilstm_mark = "✅" if bilstm_correct else ("❌" if bilstm_correct is False else "⚠️")
        bert_mark = "✅" if bert_correct else ("❌" if bert_correct is False else "⚠️")

        print(f"TEST #{i:02d} | {category:8s} | Expected: {expected:4s}")
        print(f"  Lang: {lang} (detected: {detected_lang}) | Source: {source}")
        if bilstm_score is not None:
            print(f"  Bi-LSTM: score={bilstm_score:.4f} -> {bilstm_pred:4s} {bilstm_mark}  (confidence: {abs(bilstm_score - 0.5)*200:.1f}%)")
        if bert_score is not None:
            print(f"  BERT:    score={bert_score:.4f} -> {bert_pred:4s} {bert_mark}  (confidence: {abs(bert_score - 0.5)*200:.1f}%)")
        print("-" * 70)

    # ============================================================
    # ملخص النتائج
    # ============================================================
    print("\n" + "=" * 70)
    print("  SUMMARY REPORT")
    print("=" * 70)

    categories = {
        "EN_REAL": "English Real News",
        "EN_FAKE": "English Fake News",
        "TR_REAL": "Turkish Real News",
        "TR_FAKE": "Turkish Fake News",
    }

    for cat_key, cat_name in categories.items():
        cat_results = [r for r in results if r["category"] == cat_key]
        if not cat_results:
            continue

        print(f"\n--- {cat_name} ({len(cat_results)} samples) ---")

        if bilstm_ok:
            correct = sum(1 for r in cat_results if r["bilstm_correct"])
            total = len(cat_results)
            print(f"  Bi-LSTM: {correct}/{total} correct ({correct/total*100:.0f}%)")

        if bert_ok:
            correct = sum(1 for r in cat_results if r["bert_correct"])
            total = len(cat_results)
            print(f"  BERT:    {correct}/{total} correct ({correct/total*100:.0f}%)")

    # النتيجة الإجمالية
    print(f"\n{'=' * 70}")
    print("  OVERALL ACCURACY")
    print(f"{'=' * 70}")

    if bilstm_ok:
        correct = sum(1 for r in results if r["bilstm_correct"])
        total = len(results)
        print(f"  Bi-LSTM Total: {correct}/{total} ({correct/total*100:.1f}%)")

    if bert_ok:
        correct = sum(1 for r in results if r["bert_correct"])
        total = len(results)
        print(f"  BERT Total:    {correct}/{total} ({correct/total*100:.1f}%)")

    # تحليل الأخطاء
    print(f"\n{'=' * 70}")
    print("  ERROR ANALYSIS")
    print(f"{'=' * 70}")

    for r in results:
        errors = []
        if r["bilstm_correct"] is False:
            errors.append(f"Bi-LSTM(score={r['bilstm_score']})")
        if r["bert_correct"] is False:
            errors.append(f"BERT(score={r['bert_score']})")
        if errors:
            print(f"\n  MISCLASSIFIED #{r['id']:02d} [{r['category']}] Expected={r['expected']}")
            print(f"    Failed by: {', '.join(errors)}")
            print(f"    Text: {r['text_preview']}")
            if r['detected_lang'] != r['lang']:
                print(f"    ⚠️ LANG MISMATCH: actual={r['lang']}, detected={r['detected_lang']}")

    # حفظ النتائج بصيغة JSON
    output_path = os.path.join(os.path.dirname(__file__), 'test_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n\nDetailed results saved to: {output_path}")

if __name__ == "__main__":
    main()
