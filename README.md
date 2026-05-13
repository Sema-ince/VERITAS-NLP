# VERITAS-NLP: Fake News Detection System 🤖🚩

**VERITAS-NLP**, derin öğrenme yöntemleri (BERT & Bi-LSTM) kullanarak İngilizce ve Türkçe haberlerin doğruluğunu analiz eden bir yapay zekâ projesidir. Fırat Üniversitesi Yazılım Mühendisliği Bölümü öğrencileri tarafından geliştirilmiştir.

---

## 🚀 Özellikler
- **Çift Dil Desteği:** Türkçe ve İngilizce haber analiz yeteneği.
- **Derin Öğrenme Modelleri:** 
  - **BERT (Multilingual):** Bağlamsal anlamı yakalamak için.
  - **Bi-LSTM:** Metin dizilerini her iki yönde analiz etmek için.
- **XAI (Açıklanabilir Yapay Zekâ):** LIME kullanarak modelin neden "Sahte" veya "Gerçek" dediğini kelime bazında açıklar.
- **Web Arayüzü:** Streamlit ile modern ve kullanıcı dostu dashboard.

## 🛠️ Kurulum

1. Depoyu klonlayın:
   ```bash
   git clone https://github.com/Sema-ince/VERITAS-NLP.git
   cd VERITAS-NLP
   ```

2. Gereksinimleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

3. Uygulamayı başlatın:
   ```bash
   streamlit run app.py
   ```

## 📊 Veri Seti ve Eğitim
Proje, Kaggle WELFake veri seti ve eklenen Türkçe verilerle eğitilmiştir. Veri setindeki dengesizlikler **Oversampling** ve **Stratified Splitting** yöntemleriyle giderilmiştir.

## 👥 Ekibimiz
- 👑 **Sema İnce** - Scrum Master & Arayüz Geliştirici
- 📊 **Sinan Baştuğ** - Veri Mühendisi
- 🤖 **Ahmet Al Hamed** - Yapay Zekâ Mühendisi

---
*Bu proje dezenformasyonla mücadele amacıyla akademik bir çalışma olarak hazırlanmıştır.*
