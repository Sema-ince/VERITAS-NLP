"""
VERITAS-NLP: XAI (Açıklanabilir Yapay Zekâ) Modülü
====================================================
Bu dosya, eğitilmiş modellerin kararlarını açıklamak için kullanılır.

Amaç: Model bir habere "Sahte" dediğinde, HANGİ KELİMELER yüzünden
       bu kararı verdiğini kullanıcıya göstermek.

Kullanılan Kütüphane: LIME (Local Interpretable Model-agnostic Explanations)
Çıktı: Streamlit arayüzünde gösterilebilecek kelime ağırlıkları

Örnek Çıktı:
  "şok edici"  -> +0.45 (Sahte yönünde güçlü etki)
  "Reuters"    -> -0.30 (Gerçek yönünde güçlü etki)
  "son dakika" -> +0.22 (Sahte yönünde orta etki)
"""

import os
import numpy as np
import torch
from lime.lime_text import LimeTextExplainer

# =============================================================
# 1. Bi-LSTM İÇİN TAHMİN FONKSİYONU
# =============================================================
def create_bilstm_predictor(model, vocab, max_seq_len, device):
    """
    LIME'ın kullanacağı tahmin fonksiyonunu oluşturur.
    LIME bu fonksiyona metin listesi verir, fonksiyon olasılık döndürür.
    """
    def predict_proba(texts):
        model.eval()
        results = []
        
        with torch.no_grad():
            for text in texts:
                # Metni sayılara çevir (aynı train_bilstm.py'deki gibi)
                tokens = text.split()
                encoded = [vocab.get(word, 1) for word in tokens]  # 1 = UNK
                
                # Padding
                if len(encoded) < max_seq_len:
                    encoded = encoded + [0] * (max_seq_len - len(encoded))
                else:
                    encoded = encoded[:max_seq_len]
                
                tensor = torch.tensor([encoded], dtype=torch.long).to(device)
                output = model(tensor).cpu().item()
                
                # LIME iki sınıf olasılığı ister: [P(Real), P(Fake)]
                results.append([1 - output, output])
        
        return np.array(results)
    
    return predict_proba

# =============================================================
# 2. BERT İÇİN TAHMİN FONKSİYONU
# =============================================================
def create_bert_predictor(model, tokenizer, max_length, device):
    """
    BERT modeli için LIME tahmin fonksiyonu.
    """
    def predict_proba(texts):
        model.eval()
        results = []
        
        with torch.no_grad():
            for text in texts:
                encoding = tokenizer.encode_plus(
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
                
                output = model(input_ids, attention_mask).cpu().item()
                results.append([1 - output, output])
        
        return np.array(results)
    
    return predict_proba

# =============================================================
# 3. LIME AÇIKLAMA FONKSİYONU
# =============================================================
def explain_prediction(text, predict_fn, num_features=10, num_samples=500):
    """
    Bir metin için modelin kararını açıklar.
    
    Parametreler:
    - text: Açıklanacak haber metni
    - predict_fn: Modelin tahmin fonksiyonu (create_bilstm_predictor veya create_bert_predictor)
    - num_features: Gösterilecek en etkili kelime sayısı
    - num_samples: LIME'ın test edeceği varyasyon sayısı (yüksek = daha doğru ama yavaş)
    
    Döndürür:
    - explanation_data: Her kelimenin etkisini gösteren sözlük listesi
    - predicted_class: Modelin tahmini (0=Real, 1=Fake)
    - confidence: Modelin güven oranı
    """
    # LIME açıklayıcısını oluştur
    explainer = LimeTextExplainer(class_names=["Real (Gerçek)", "Fake (Sahte)"])
    
    # Açıklama oluştur
    explanation = explainer.explain_instance(
        text,
        predict_fn,
        num_features=num_features,
        num_samples=num_samples
    )
    
    # Tahmin sonuçlarını al
    probs = predict_fn([text])[0]
    predicted_class = int(probs[1] > 0.5)  # 0=Real, 1=Fake
    confidence = max(probs) * 100
    
    # Kelime ağırlıklarını düzenle
    feature_weights = explanation.as_list()
    explanation_data = []
    for word, weight in feature_weights:
        explanation_data.append({
            'word': word,
            'weight': round(weight, 4),
            'direction': 'Sahte 🔴' if weight > 0 else 'Gerçek 🟢',
            'impact': 'Güçlü' if abs(weight) > 0.1 else 'Orta' if abs(weight) > 0.05 else 'Zayıf'
        })
    
    return {
        'explanation': explanation_data,
        'predicted_class': predicted_class,
        'predicted_label': 'Fake (Sahte)' if predicted_class == 1 else 'Real (Gerçek)',
        'confidence': round(confidence, 2),
        'lime_explanation_object': explanation  # Streamlit'te HTML görselleştirme için
    }

# =============================================================
# 4. STREAMLIT İÇİN HAZIR FONKSİYON
# =============================================================
def get_explanation_html(explanation_result):
    """
    LIME açıklama sonucunu Streamlit'te gösterilebilecek HTML formatına çevirir.
    """
    lime_exp = explanation_result.get('lime_explanation_object')
    if lime_exp:
        return lime_exp.as_html()
    return "<p>Açıklama oluşturulamadı.</p>"


def format_explanation_for_display(explanation_result):
    """
    Açıklama sonucunu Streamlit'te tablo/metin olarak göstermek için düzenler.
    
    Kullanım (app.py içinde):
        result = explain_prediction(text, predict_fn)
        summary, details = format_explanation_for_display(result)
        st.write(summary)
        st.dataframe(details)
    """
    pred_label = explanation_result['predicted_label']
    confidence = explanation_result['confidence']
    
    summary = f"🔍 **Model Kararı:** {pred_label} (Güven: %{confidence})"
    
    details = []
    for item in explanation_result['explanation']:
        details.append({
            'Kelime': item['word'],
            'Ağırlık': item['weight'],
            'Yön': item['direction'],
            'Etki Gücü': item['impact']
        })
    
    return summary, details


# =============================================================
# 5. TEST (Tek başına çalıştırılırsa)
# =============================================================
if __name__ == "__main__":
    print("=" * 60)
    print(" VERITAS-NLP: XAI Modülü Test")
    print("=" * 60)
    print("\nBu dosya doğrudan çalıştırılmak için değil,")
    print("app.py veya diğer scriptler tarafından içe aktarılmak (import) için tasarlanmıştır.")
    print("\nKullanım örneği:")
    print("  from scripts.xai_explainer import explain_prediction, create_bert_predictor")
    print("  predict_fn = create_bert_predictor(model, tokenizer, max_length, device)")
    print("  result = explain_prediction('Haber metni buraya...', predict_fn)")
    print("  print(result['predicted_label'])")
    print("  print(result['explanation'])")
