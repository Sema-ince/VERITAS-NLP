import streamlit as st
import requests
from bs4 import BeautifulSoup
import os
import torch
import pandas as pd

# =============================================================
# MODEL YÜKLEME FONKSİYONLARI
# =============================================================

@st.cache_resource
def load_bilstm_model():
    """Eğitilmiş Bi-LSTM modelini yükler (Ahmet'in modeli)."""
    from scripts.train_bilstm import BiLSTMClassifier
    
    model_path = os.path.join("models", "saved", "bilstm_model.pt")
    if not os.path.exists(model_path):
        return None, None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    
    hp = checkpoint['hyperparameters']
    model = BiLSTMClassifier(
        vocab_size=hp['vocab_size'],
        embedding_dim=hp['embedding_dim'],
        hidden_size=hp['hidden_size'],
        num_layers=hp['num_layers'],
        dropout=hp['dropout']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    vocab = checkpoint['vocab_word2idx']
    return model, vocab

@st.cache_resource
def load_bert_model():
    """Eğitilmiş BERT modelini yükler (Ahmet'in modeli)."""
    from scripts.train_bert import BertClassifier
    from transformers import BertTokenizer
    
    model_path = os.path.join("models", "saved", "bert_model.pt")
    if not os.path.exists(model_path):
        return None, None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    
    model = BertClassifier(
        model_name=checkpoint['model_name'],
        dropout=checkpoint['dropout']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    tokenizer = BertTokenizer.from_pretrained(checkpoint['model_name'])
    return model, tokenizer

# =============================================================
# TAHMİN FONKSİYONLARI
# =============================================================

def predict_with_bilstm(text, model, vocab, max_seq_len=256):
    """Bi-LSTM modeli ile tahmin yapar."""
    device = next(model.parameters()).device
    tokens = text.split()
    encoded = [vocab.get(word, 1) for word in tokens]
    
    if len(encoded) < max_seq_len:
        encoded = encoded + [0] * (max_seq_len - len(encoded))
    else:
        encoded = encoded[:max_seq_len]
    
    tensor = torch.tensor([encoded], dtype=torch.long).to(device)
    
    with torch.no_grad():
        output = model(tensor).cpu().item()
    
    return output  # 0'a yakın = Real, 1'e yakın = Fake

def predict_with_bert(text, model, tokenizer, max_length=128):
    """BERT modeli ile tahmin yapar."""
    device = next(model.parameters()).device
    
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
    
    with torch.no_grad():
        output = model(input_ids, attention_mask).cpu().item()
    
    return output  # 0'a yakın = Real, 1'e yakın = Fake

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
        
        st.markdown("---")
        st.subheader("🧠 Açıklanabilir Yapay Zekâ (XAI) Analizi")
        st.markdown(summary)
        
        if details:
            st.markdown("**Modelin kararında en etkili kelimeler:**")
            df = pd.DataFrame(details)
            st.dataframe(df, use_container_width=True, hide_index=True)

# =============================================================
# 1. Sayfa Ayarları ve Sekme Başlığı
# =============================================================
st.set_page_config(page_title="VERITAS-NLP", page_icon="📰", layout="wide")

# --- İNGİLİZCE MENÜYÜ VE YAZILARI GİZLEME KODU ---
gizleme_stili = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(gizleme_stili, unsafe_allow_html=True)
# ------------------------------------------------

# 2. Sol Menü (Sidebar) Oluşturma
st.sidebar.title("Menü")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2102/2102117.png", width=100) 
sayfa = st.sidebar.radio("Sayfa Seçiniz:", ["Metin Girişi (Analiz)", "Hakkımızda"])

# --- Model durumu sidebar'da göster ---
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Model Durumu")

bilstm_model, bilstm_vocab = load_bilstm_model()
bert_model, bert_tokenizer = load_bert_model()

if bilstm_model:
    st.sidebar.success("✅ Bi-LSTM modeli yüklü")
else:
    st.sidebar.warning("⏳ Bi-LSTM modeli henüz eğitilmedi")

if bert_model:
    st.sidebar.success("✅ BERT modeli yüklü")
else:
    st.sidebar.warning("⏳ BERT modeli henüz eğitilmedi")

# =============================================================
# 3. Sayfa Yönlendirmeleri
# =============================================================
if sayfa == "Metin Girişi (Analiz)":
    st.title("📰 VERITAS-NLP: Sahte Haber Tespit Sistemi")
    st.write("Lütfen analiz etmek istediğiniz haber metnini veya linkini aşağıya girin.")
    
    # Kullanıcıya iki seçenek sunmak için sekmeler oluşturuyoruz
    tab1, tab2 = st.tabs(["📝 Metin Yapıştır", "🔗 Haber Linki (URL) Gir"])
    
    with tab1:
        # Metin girme kutusu
        haber_metni = st.text_area("Haber Metni:", height=200, placeholder="Haber metnini buraya yapıştırın...")
        if st.button("Metni Analiz Et", key="analyze_text"):
            if haber_metni:
                # Model kontrolü
                models_available = bilstm_model is not None or bert_model is not None
                
                if not models_available:
                    st.warning("⚠️ Henüz eğitilmiş model bulunamadı. Lütfen önce modelleri eğitin.")
                    st.info("Komutlar:\n- `python scripts/train_bilstm.py`\n- `python scripts/train_bert.py`")
                else:
                    st.markdown("---")
                    st.subheader("📊 Analiz Sonuçları")
                    
                    # Bi-LSTM ile tahmin
                    if bilstm_model:
                        score = predict_with_bilstm(haber_metni, bilstm_model, bilstm_vocab)
                        show_result(score, "Bi-LSTM")
                    
                    # BERT ile tahmin
                    if bert_model:
                        score = predict_with_bert(haber_metni, bert_model, bert_tokenizer)
                        is_fake, conf = show_result(score, "BERT")
                        
                        # XAI açıklaması (BERT modeli için)
                        from scripts.xai_explainer import create_bert_predictor
                        device = next(bert_model.parameters()).device
                        predict_fn = create_bert_predictor(bert_model, bert_tokenizer, 128, device)
                        show_xai_explanation(haber_metni, predict_fn)
                    
                    elif bilstm_model:
                        # BERT yoksa Bi-LSTM için XAI
                        from scripts.xai_explainer import create_bilstm_predictor
                        device = next(bilstm_model.parameters()).device
                        predict_fn = create_bilstm_predictor(bilstm_model, bilstm_vocab, 256, device)
                        show_xai_explanation(haber_metni, predict_fn)
            else:
                st.warning("Lütfen analiz etmek için bir metin girin!")
                
    with tab2:
        # Link girme kutusu
        haber_linki = st.text_input("Haber Linki (URL):", placeholder="Örn: https://www.hurriyet.com.tr/...")
        
        if st.button("Linkten Analiz Et", key="analyze_url"):
            if haber_linki:
                with st.spinner('Haber metni çekiliyor...'):
                    try:
                        # 1. Sitenin bizi robot sanıp engellememesi için tarayıcı kılığına giriyoruz
                        headers = {'User-Agent': 'Mozilla/5.0'}
                        
                        # 2. Linke gidip sitenin kodlarını alıyoruz
                        cevap = requests.get(haber_linki, headers=headers)
                        soup = BeautifulSoup(cevap.text, 'html.parser')
                        
                        # 3. Sitedeki tüm paragrafları (<p> etiketleri) bulup birleştiriyoruz
                        paragraflar = soup.find_all('p')
                        cekilen_metin = " ".join([p.text for p in paragraflar])
                        
                        # 4. Eğer anlamlı bir metin bulabildiysek
                        if len(cekilen_metin) > 100:
                            st.success("Haber metni başarıyla çekildi!")
                            
                            st.markdown("**Çekilen Metin (Önizleme):**")
                            st.info(cekilen_metin[:500] + " ... (Devamı var)")
                            
                            # Model ile analiz
                            models_available = bilstm_model is not None or bert_model is not None
                            
                            if models_available:
                                st.markdown("---")
                                st.subheader("📊 Analiz Sonuçları")
                                
                                if bilstm_model:
                                    score = predict_with_bilstm(cekilen_metin, bilstm_model, bilstm_vocab)
                                    show_result(score, "Bi-LSTM")
                                
                                if bert_model:
                                    score = predict_with_bert(cekilen_metin, bert_model, bert_tokenizer)
                                    show_result(score, "BERT")
                                    
                                    from scripts.xai_explainer import create_bert_predictor
                                    device = next(bert_model.parameters()).device
                                    predict_fn = create_bert_predictor(bert_model, bert_tokenizer, 128, device)
                                    show_xai_explanation(cekilen_metin, predict_fn)
                            else:
                                st.warning("⚠️ Henüz eğitilmiş model bulunamadı.")
                        else:
                            st.warning("Bu linkten yeterli metin çekilemedi. Sitenin yapısı farklı olabilir, başka bir haber linki deneyin.")
                            
                    except Exception as e:
                        st.error(f"Link çekilirken bir hata oluştu: {e}")
            else:
                st.warning("Lütfen bir haber linki girin!")

elif sayfa == "Hakkımızda":
# ... (Hakkımızda kısmı aynı kalacak) ...
    st.title("👥 Hakkımızda")
    st.write("Bu proje Fırat Üniversitesi Yazılım Mühendisliği Bölümü öğrencileri tarafından geliştirilmektedir.")
    
    st.subheader("Proje Üyeleri")
    st.write("👑 **Sema İnce** - Scrum Master & Arayüz Geliştirici")
    st.write("📊 **Sinan Baştuğ** - Veri Mühendisi")
    st.write("🤖 **Ahmet Al Hamed** - Yapay Zekâ Mühendisi")
    
    st.write("---")
    st.write("**Amacımız:** Derin öğrenme (BERT & Bi-LSTM) ve XAI yöntemleri kullanarak dezenformasyonla mücadele etmektir.")