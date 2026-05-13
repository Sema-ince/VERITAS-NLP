import streamlit as st
import requests
from bs4 import BeautifulSoup
import time

import feedparser # RSS için gerekli (pip install feedparser)

# Ahmet'in yazdığı XAI (Açıklanabilirlik) fonksiyonlarını arayüze dahil ediyoruz
import sys
import os

# scripts klasöründeki dosyalara erişebilmek için dosya yolunu ekliyoruz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts')))

try:
    from scripts.xai_explainer import explain_prediction, format_explanation_for_display, get_explanation_html
    XAI_HAZIR = True
except ImportError:
    st.warning("XAI modülü bulunamadı. Lütfen 'scripts' klasöründe 'xai_explainer.py' olduğundan emin olun.")
    XAI_HAZIR = False
# ==========================================
# 📌 1. KISIM: AHMET'İN MODELLERİ (YÜKLEME)
# ==========================================
import torch
from scripts.train_bilstm import BiLSTMClassifier

@st.cache_resource
def load_bilstm():
    path = "models/saved/bilstm_model.pt"
    if not os.path.exists(path):
        return None, None
    try:
        # إجبار التحميل على المعالج لضمان عدم استهلاك VRAM إضافي
        checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
        hp = checkpoint['hyperparameters']
        model = BiLSTMClassifier(hp['vocab_size'], hp['embedding_dim'], hp['hidden_size'], hp['num_layers'], hp['dropout'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, checkpoint['vocab_word2idx']
    except Exception as e:
        st.error(f"⚠️ Bi-LSTM yükleme hatası: {e}")
        return None, None

@st.cache_resource
def load_bert():
    from scripts.train_bert import BertClassifier
    from transformers import BertTokenizer
    path = "models/saved/bert_model.pt"
    if not os.path.exists(path):
        return None, None
    try:
        checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
        model_name = checkpoint.get('model_name', "bert-base-multilingual-cased")
        dropout = checkpoint.get('dropout', 0.3)
        
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertClassifier(model_name, dropout)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"⚠️ BERT yükleme hatası: {e}")
        return None, None

bilstm_model, bilstm_vocab = load_bilstm()
bert_model, bert_tokenizer = load_bert()

# =============================================================
# TAHMİN FONKSİYONLARI
# =============================================================

def detect_language(text):
    """Metnin dilini basit bir sezgisel yöntemle algılar."""
    # Türkçeye özgü karakterler
    turkish_chars = set('çşğüöıÇŞĞÜÖİ')
    text_chars = set(text)
    if text_chars & turkish_chars:
        return 'tr'
    
    # Türkçeye özgü yaygın kelimeler
    turkish_words = {'bir', 've', 'ile', 'olan', 'için', 'bu', 'da', 'de',
                     'den', 'dan', 'olarak', 'ise', 'gibi', 'daha', 'sonra',
                     'kadar', 'gore', 'icin', 'ile', 'hem', 'ama', 'ancak',
                     'yarin', 'bugun', 'haber', 'haberi', 'acikladi', 'soyledi',
                     'turkiye', 'istanbul', 'ankara', 'devlet', 'bakani',
                     'cumhurbaskani', 'hukumet', 'basvurun', 'vatandaslarina',
                     'dagitacagini', 'buyudu', 'oraninda', 'kuruldu', 'ilinin',
                     'beylik', 'devleti', 'imparatorlugu', 'fethedip', 'vererek'}
    words = set(text.lower().split())
    turkish_match = len(words & turkish_words)
    if turkish_match >= 2:
        return 'tr'
    
    return 'en'

def preprocess_text_for_bilstm(text):
    """Metni Bi-LSTM modeli için ön işler (eğitimdeki gibi, birleşik pipeline)."""
    import re
    if not text:
        return ""
    # Küçük harf
    text = text.lower()
    # Sadece harfleri koru (İngilizce + Türkçe)
    text = re.sub(r'[^a-zçşğüöı\s]', ' ', text)
    # Fazla boşlukları temizle
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_with_bilstm(text, model, vocab, max_seq_len=256):
    """Bi-LSTM modeli ile tahmin yapar."""
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
    
    return output  # 0'a yakın = Real, 1'e yakın = Fake

def predict_with_bert(text, model, tokenizer, max_length=128):
    """BERT modeli ile tahmin yapar."""
    device = next(model.parameters()).device

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

    return output  # 0'a yakın = Real, 1'e yakın = Fake

def predict_ensemble(text):
    """İki modeli birleştirerek daha iyi tahmin yapar (Weighted Ensemble)."""
    bilstm_score = predict_with_bilstm(text, bilstm_model, bilstm_vocab) if bilstm_model else None
    bert_score = predict_with_bert(text, bert_model, bert_tokenizer) if bert_model else None

    if bilstm_score is None and bert_score is None:
        return None
    if bilstm_score is None:
        return bert_score
    if bert_score is None:
        return bilstm_score

    # BERT daha başarılı olduğu için (75% vs 55%) daha yüksek ağırlık
    bert_weight = 0.65
    bilstm_weight = 0.35
    return bert_weight * bert_score + bilstm_weight * bilstm_score

# =============================================================
# SONUÇ GÖSTERME FONKSİYONU
# =============================================================

def show_result(score, model_name):
    """Tahmin sonucunu ekranda gösterir."""
    is_fake = score > 0.5
    confidence = score * 100 if is_fake else (1 - score) * 100
    
    if is_fake:
        st.error(f"🔴 **{model_name} Sonucu:** Bu haber **SAHTE** olarak tespit edildi! (Güven: %{confidence:.1f})")
    else:
        st.success(f"🟢 **{model_name} Sonucu:** Bu haber **GERÇEK** olarak tespit edildi. (Güven: %{confidence:.1f})")
    
    return is_fake, confidence

def show_xai_explanation(text, predict_fn):
    """XAI açıklamasını gösterir."""
    from scripts.xai_explainer import explain_prediction, format_explanation_for_display
    
    with st.spinner("🔍 Model kararı açıklanıyor (XAI analizi)..."):
        result = explain_prediction(text, predict_fn, num_features=10, num_samples=300)
        summary, details = format_explanation_for_display(result)
        
        # Sonuçları ekranda göster
        st.subheader("🧠 Model Bu Kararı Neden Verdi? (XAI Analizi)")
        st.write(summary)
        if details:
            import pandas as pd
            st.dataframe(pd.DataFrame(details), use_container_width=True)

def haber_analiz_et(metin):
    """URL ve RSS'ten gelen haberleri gerçek modellerle analiz eder."""
    # Model kontrolü
    models_available = (bilstm_model is not None) or (bert_model is not None)

    if not models_available:
        st.warning("⚠️ Henüz eğitilmiş model bulunamadı. Lütfen önce modelleri eğitin.")
        return

    st.markdown("---")
    st.subheader("📊 Analiz Sonuçları")

    # Dil algılama
    detected_lang = detect_language(metin)
    if detected_lang == 'tr':
        st.info("🌐 **Dil Algılama:** Türkçe metin tespit edildi.")

    # Ensemble (her iki model varsa)
    if bilstm_model is not None and bert_model is not None:
        st.markdown("**🤖 Ensemble Prediction (İki Modelin Birleşimi):**")
        ensemble_score = predict_ensemble(metin)
        if ensemble_score is not None:
            is_fake, conf = show_result(ensemble_score, "ENSEMBLE")

        # XAI açıklaması
        if XAI_HAZIR:
            st.markdown("---")
            from scripts.xai_explainer import create_bert_predictor
            device = next(bert_model.parameters()).device
            predict_fn = create_bert_predictor(bert_model, bert_tokenizer, 128, device)
            show_xai_explanation(metin, predict_fn)

    # Sadece bir model varsa
    elif bert_model is not None:
        score = predict_with_bert(metin, bert_model, bert_tokenizer)
        is_fake, conf = show_result(score, "BERT")

        if XAI_HAZIR:
            from scripts.xai_explainer import create_bert_predictor
            device = next(bert_model.parameters()).device
            predict_fn = create_bert_predictor(bert_model, bert_tokenizer, 128, device)
            show_xai_explanation(metin, predict_fn)

    elif bilstm_model is not None:
        score = predict_with_bilstm(metin, bilstm_model, bilstm_vocab)
        show_result(score, "Bi-LSTM")

        if XAI_HAZIR:
            from scripts.xai_explainer import create_bilstm_predictor
            device = next(bilstm_model.parameters()).device
            predict_fn = create_bilstm_predictor(bilstm_model, bilstm_vocab, 256, device)
            show_xai_explanation(metin, predict_fn)

# ==========================================
# 📌 3. KISIM: ARAYÜZ VE SAYFA TASARIMI
# ==========================================
st.set_page_config(page_title="VERITAS-NLP", page_icon="📰", layout="wide")

gizleme_stili = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(gizleme_stili, unsafe_allow_html=True)

st.sidebar.title("Menü")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2102/2102117.png", width=100) 
sayfa = st.sidebar.radio("Sayfa Seçiniz:", ["Metin Girişi (Analiz)", "Hakkımızda"])

if sayfa == "Metin Girişi (Analiz)":
    st.title("📰 VERITAS-NLP: Sahte Haber Tespit Sistemi")
    st.write("Lütfen analiz etmek istediğiniz haber metnini veya linkini aşağıya girin.")
    
    # Yeni planımıza uygun 3 sekmeli yapı
    tab1, tab2, tab3 = st.tabs(["📝 Metin Yapıştır", "🔗 Haber Linki (URL)", "📡 Canlı RSS Akışı"])
    
    # 1. MODÜL: MANUEL GİRİŞ
    with tab1:
        haber_metni = st.text_area("Haber Metni:", height=200, placeholder="Haber metnini buraya yapıştırın...")
        if st.button("Metni Analiz Et", key="btn_manuel"):
            if haber_metni and len(haber_metni) > 50:
                # Model kontrolü
                models_available = ('bilstm_model' in globals() and bilstm_model is not None) or \
                                   ('bert_model' in globals() and bert_model is not None)
                
                if not models_available:
                    st.warning("⚠️ Henüz eğitilmiş model bulunamadı. Lütfen önce modelleri eğitin.")
                else:
                    st.markdown("---")
                    st.subheader("📊 Analiz Sonuçları")

                    # Dil algılama
                    detected_lang = detect_language(haber_metni)
                    if detected_lang == 'tr':
                        st.info("🌐 **Dil Algılama:** Türkçe metin tespit edildi.")

                    # Ensemble kullan (her iki model varsa)
                    if 'bilstm_model' in globals() and bilstm_model and 'bert_model' in globals() and bert_model:
                        st.markdown("**🤖 Ensemble Prediction (İki Modelin Birleşimi):**")
                        ensemble_score = predict_ensemble(haber_metni)
                        if ensemble_score is not None:
                            is_fake, conf = show_result(ensemble_score, "ENSEMBLE")

                        # XAI
                        st.markdown("---")
                        from scripts.xai_explainer import create_bert_predictor
                        device = next(bert_model.parameters()).device
                        predict_fn = create_bert_predictor(bert_model, bert_tokenizer, 128, device)
                        show_xai_explanation(haber_metni, predict_fn)

                    else:
                        # Sadece bir model varsa
                        if 'bert_model' in globals() and bert_model:
                            score = predict_with_bert(haber_metni, bert_model, bert_tokenizer)
                            is_fake, conf = show_result(score, "BERT")

                            # XAI
                            from scripts.xai_explainer import create_bert_predictor
                            device = next(bert_model.parameters()).device
                            predict_fn = create_bert_predictor(bert_model, bert_tokenizer, 128, device)
                            show_xai_explanation(haber_metni, predict_fn)

                        elif 'bilstm_model' in globals() and bilstm_model:
                            score = predict_with_bilstm(haber_metni, bilstm_model, bilstm_vocab)
                            show_result(score, "Bi-LSTM")

                            from scripts.xai_explainer import create_bilstm_predictor
                            device = next(bilstm_model.parameters()).device
                            predict_fn = create_bilstm_predictor(bilstm_model, bilstm_vocab, 256, device)
                            show_xai_explanation(haber_metni, predict_fn)
            else:
                st.warning("Lütfen analiz etmek için anlamlı bir metin (en az 50 karakter) girin!")
                
    # 2. MODÜL: URL KAZIMA
    with tab2:
        haber_linki = st.text_input("Haber Linki (URL):", placeholder="Örn: https://www.hurriyet.com.tr/...")
        if st.button("Linkten Analiz Et", key="btn_url"):
            if haber_linki:
                try:
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    cevap = requests.get(haber_linki, headers=headers)
                    soup = BeautifulSoup(cevap.text, 'html.parser')
                    paragraflar = soup.find_all('p')
                    cekilen_metin = " ".join([p.text for p in paragraflar])
                    
                    if len(cekilen_metin) > 100:
                        st.info("Haber metni başarıyla çekildi. Analiz ediliyor...")
                        haber_analiz_et(cekilen_metin)
                    else:
                        st.warning("Bu linkten yeterli metin çekilemedi.")
                except Exception as e:
                    st.error(f"Link çekilirken bir hata oluştu: {e}")
            else:
                st.warning("Lütfen bir haber linki girin!")

    # 3. MODÜL: RSS AKIŞI
    with tab3:
        st.info("Güvenilir kaynaklardan (Örn: TRT Haber, NTV) canlı haber akışı sağlanıyor.")
        
        # Güncellenmiş TRT Haber RSS linki
        rss_url = "https://www.trthaber.com/sondakika_articles.rss" 
        
        # Sitenin bizi robot sanıp engellememesi için tarayıcı kılığına (User-Agent) giriyoruz
        feed = feedparser.parse(rss_url, agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64)')
        
        if feed.entries:
            # RSS'ten gelen ilk 5 haberi alıp başlıklarını listeliyoruz
            haber_secenekleri = {entry.title: entry.description for entry in feed.entries[:5]}
            secilen_baslik = st.selectbox("Analiz edilecek canlı haberi seçin:", list(haber_secenekleri.keys()))
            
            # Seçilen haberin özetini ekrana yazdır
            st.write("**Haber Özeti:**", haber_secenekleri[secilen_baslik])
            
            if st.button("Seçili Canlı Haberi Analiz Et", key="btn_rss"):
                haber_analiz_et(haber_secenekleri[secilen_baslik])
        else:
            st.error("RSS akışı şu an alınamıyor. Lütfen bağlantınızı kontrol edin.")
elif sayfa == "Hakkımızda":
    st.title("👥 Hakkımızda")
    st.write("Bu proje Fırat Üniversitesi Yazılım Mühendisliği Bölümü öğrencileri tarafından geliştirilmektedir.")
    
    st.subheader("Proje Üyeleri")
    st.write("👑 **Sema İnce** - Scrum Master & Arayüz Geliştirici")
    st.write("📊 **Sinan Baştuğ** - Veri Mühendisi")
    st.write("🤖 **Ahmet Al Hamed** - Yapay Zekâ Mühendisi")
    
    st.write("---")
    st.write("**Amacımız:** Derin öğrenme (BERT & Bi-LSTM) ve XAI yöntemleri kullanarak dezenformasyonla mücadele etmektir.")