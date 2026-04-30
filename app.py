import streamlit as st
import requests
from bs4 import BeautifulSoup
import time
import random
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
@st.cache_resource # Modelin her sayfada baştan yüklenmesini engeller (Hızlandırır)
def modeli_yukle():
    # TODO: Ahmet'in modeli buraya gelecek. Örnek:
    # model = tf.keras.models.load_model('veritas_bilstm_model.h5')
    # return model
    return "MOCK_MODEL" # Şimdilik taklit model dönüyoruz

yapay_zeka_modeli = modeli_yukle()

# ==========================================
# 📌 2. KISIM: MERKEZİ ANALİZ VE XAI FONKSİYONU
# ==========================================
def haber_analiz_et(metin):
    # Kullanıcıya analiz yapıldığını hissettiren animasyon
    with st.spinner("Yapay Zeka (Bi-LSTM & BERT) metni analiz ediyor..."):
        time.sleep(1.5) # Gerçekçilik katmak için 1.5 saniye bekleme
        
        # TODO: Ahmet'in predict (tahmin) kodu buraya gelecek. Örn:
        # sonuc = yapay_zeka_modeli.predict(temizlenmis_metin)
        
        # Şimdilik rastgele bir sahtelik oranı üretiyoruz (Mock)
        sahtelik_orani = random.randint(10, 95) 
        
        # 3. MADDENİN ÇÖZÜMÜ: Yüzdelik Oranın Görselleştirilmesi
        st.subheader("📊 Analiz Sonucu")
        
        # İki kolon oluşturup görseli zenginleştiriyoruz
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.metric(label="Sahte Olma İhtimali", value=f"%{sahtelik_orani}")
            
        with col2:
            # Orana göre bar çubuğunun rengini belirliyoruz
            if sahtelik_orani > 70:
                st.error("Bu haberin SAHTE (Dezenformasyon) olma ihtimali çok yüksek!")
                st.progress(sahtelik_orani)
            elif sahtelik_orani > 40:
                st.warning("Bu haber ŞÜPHELİ. Dikkatli yaklaşılmalı.")
                st.progress(sahtelik_orani)
            else:
                st.success("Bu haber büyük ihtimalle GERÇEK.")
                st.progress(sahtelik_orani)
        
        st.write("---")
        
        # 4. MADDENİN ÇÖZÜMÜ: XAI (Açıklanabilir Yapay Zeka) Görselleştirmesi
        st.subheader("🧠 Model Bu Kararı Neden Verdi? (XAI Analizi)")
        st.write("Modelin kararını etkileyen anahtar kelimeler aşağıda vurgulanmıştır:")
        
        # XAI Taklit Çıktısı (Ahmet SHAP/LIME entegre edince bu html yapısına kelimeleri gönderecek)
        # Kırmızı: Sahteliği artıranlar, Yeşil: Gerçekliği artıranlar
        xai_ornek_cikti = f"""
        <div style="padding:10px; border-radius:5px; background-color:#1e1e1e; line-height: 2;">
            Bu haberdeki <span style="background-color:#ff4b4b; color:white; padding:3px; border-radius:3px;">şok edici</span> 
            iddialara göre yetkililer <span style="background-color:#4baf4b; color:white; padding:3px; border-radius:3px;">resmi bir açıklama</span> 
            yapmaktan kaçındı. Olayın <span style="background-color:#ff4b4b; color:white; padding:3px; border-radius:3px;">gizli belgeleri</span> 
            sızdırıldı.
        </div>
        """
        st.markdown(xai_ornek_cikti, unsafe_allow_html=True)

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
                haber_analiz_et(haber_metni)
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