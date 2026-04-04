import streamlit as st
import requests
from bs4 import BeautifulSoup

# 1. Sayfa Ayarları ve Sekme Başlığı
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

# ... (KODUN GERİ KALANI AYNI ŞEKİLDE DURACAK) ...

# 3. Sayfa Yönlendirmeleri
# 3. Sayfa Yönlendirmeleri
if sayfa == "Metin Girişi (Analiz)":
    st.title("📰 VERITAS-NLP: Sahte Haber Tespit Sistemi")
    st.write("Lütfen analiz etmek istediğiniz haber metnini veya linkini aşağıya girin.")
    
    # Kullanıcıya iki seçenek sunmak için sekmeler oluşturuyoruz
    tab1, tab2 = st.tabs(["📝 Metin Yapıştır", "🔗 Haber Linki (URL) Gir"])
    
    with tab1:
        # Metin girme kutusu
        haber_metni = st.text_area("Haber Metni:", height=200, placeholder="Haber metnini buraya yapıştırın...")
        if st.button("Metni Analiz Et"):
            if haber_metni:
                st.success("Metin alındı! (Arka planda Ahmet'in modeli buraya bağlanacak)")
            else:
                st.warning("Lütfen analiz etmek için bir metin girin!")
                
    with tab2:
        # Link girme kutusu
        haber_linki = st.text_input("Haber Linki (URL):", placeholder="Örn: https://www.hurriyet.com.tr/...")
        
        if st.button("Linkten Analiz Et"):
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
                        
                        # 4. Eğer anlamlı bir metin bulabildiysek ekrana yazdırıyoruz
                        if len(cekilen_metin) > 100:
                            st.success("Haber metni başarıyla çekildi!")
                            
                            st.markdown("**Arka Planda Çekilen Metin:**")
                            # Ekranda çok yer kaplamaması için ilk 500 karakteri gösteriyoruz
                            st.info(cekilen_metin[:500] + " ... (Devamı var)") 
                            
                            st.write("*(İşte bu çekilen tam metin arka planda Ahmet'in modeline yollanacak!)*")
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