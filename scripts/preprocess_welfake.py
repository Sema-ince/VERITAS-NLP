import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

# Etkileşimli (interaktif) projelere tqdm progress bar desteği
tqdm.pandas()

def setup_nltk():
    """NLTK kütüphanesi için gerekli veri dosyalarını indirir."""
    print("NLTK veri dosyaları kontrol ediliyor/indiriliyor...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    # Punkt tokenizer ve kelime ağını kullanabilmek için ek veriler bazı sürümlerde gerekebilir:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print("NLTK dosyaları hazır.\n")

def clean_text(text, lemmatizer, stop_words):
    """
    Metin NLP ön işleme adımları (Lemmatization ile):
    1. Lowercase (Küçük harfe çevirme)
    2. Noktalama, özel karakterler ve sayıları kaldırma (Sadece harfler kalsın)
    3. Tokenization (Kelime kelime ayırma)
    4. Stopwords kaldırma
    5. Lemmatize (Kök/Gövde bulma)
    """
    # Eksik/Boş veriler için güvenlik kontrolü
    if pd.isna(text):
        return ""
    
    # 1. Küçük harf
    text = str(text).lower()
    
    # 2. Noktalama, rakam ve özel karakterleri atma (Sadece a-z arası kalır)
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # 3. Fazla boşlukları temizleme
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Boş kalırsa hemen dönelim
    if not text:
        return ""
    
    # Tokenization & Stopwords & Lemmatization aynı döngüde performans için
    tokens = text.split()
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return ' '.join(cleaned_tokens)

def main():
    print("="*50)
    print(" VERITAS-NLP: WELFake Veri Seti Ön İşleme")
    print("="*50 + "\n")
    
    # NLTK gereksinimlerini kontrol et
    setup_nltk()

    # Dosya yolları
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(current_dir, "..", "data", "raw", "WELFake_Dataset.csv")
    processed_dir = os.path.join(current_dir, "..", "data", "processed")
    processed_data_path = os.path.join(processed_dir, "WELFake_cleaned.csv")
    
    # İşlenmiş veri klasörü yoksa oluştur
    os.makedirs(processed_dir, exist_ok=True)
    
    if not os.path.exists(raw_data_path):
        print(f"HATA: Ham veri seti bulunamadı -> {raw_data_path}")
        return

    print(f"Veri seti yükleniyor... ({os.path.basename(raw_data_path)})")
    df = pd.read_csv(raw_data_path)
    
    initial_shape = df.shape
    print(f"Veri başarıyla yüklendi. Dosya boyutu: {initial_shape}\n")
    
    print("Adım 1: Eksik Verileri Temizleme/Yönetme...")
    # Haberlerde hem title hem text birleştirileceği için eksik olanları boş string yapıyoruz
    df['title'].fillna("", inplace=True)
    df['text'].fillna("", inplace=True)
    
    print("Adım 2: Başlık ('title') ve Metin ('text') Sütunlarını Birleştirme...")
    df['content'] = df['title'] + " " + df['text']
    
    # Hafızayı rahatlatmak için 'title' ve 'text' düşürülebilir, orijinal ID numarası vb. durabilir
    # (İsteğe bağlı olarak bu adım iptal edilebilir, ama NLP eğitiminde genelde combined content kullanılır)
    df.drop(columns=['title', 'text'], inplace=True, errors='ignore')

    # Başlığın ardından gelen fazla boşlukları düzeltmek için
    df['content'] = df['content'].str.strip()

    print("\nAdım 3: Metin Ön İşleme (Temizleme & Lemmatization) Uygulanıyor...")
    print("Bu işlem satır sayısına bağlı olarak birkaç dakika sürebilir (Örn: 72.000 satır). Lütfen bekleyin...")
    
    lemmatizer = WordNetLemmatizer()
    # İngilizce stop words listesini alıp daha hızlı arama için set'e dönüştürüyoruz
    stop_words = set(stopwords.words('english'))
    
    # progress_apply (tqdm.pandas üzerinden gelir) sadece apply'ın ilerleme çubuklu halidir
    df['content'] = df['content'].progress_apply(lambda x: clean_text(x, lemmatizer, stop_words))
    
    # Ekstra kontrol: Temizledikten sonra içerik tamamen silindiyse silebiliriz:
    empty_content_mask = df['content'] == ""
    if empty_content_mask.sum() > 0:
        print(f"\nUyarı: {empty_content_mask.sum()} kaydın içi tamamen boşaldı, veri setinden atılıyor...")
        df = df[~empty_content_mask].reset_index(drop=True)

    print("\nAdım 4: İşlenmiş Veriyi Kaydetme...")
    print(f"Oluşturuluyor: {processed_data_path}")
    
    # Sonucu CSV'ye yaz (satır indeksini kaydetme)
    # NLP modellerinde dosya boyutundan tasarruf için gzip ile sıkıştırılabilir, ama şimdilik csv
    df.to_csv(processed_data_path, index=False)
    
    final_shape = df.shape
    print(f"\nİşlem Tamamlandı!")
    print(f"-> İlk Durum ({initial_shape[0]} satır, {initial_shape[1]} sütun)")
    print(f"-> Son Durum ({final_shape[0]} satır, {final_shape[1]} sütun)")
    
    print("\nVeri artık eğitime hazır (Week 2 Görevleri başarıyla tamamlandı).")

if __name__ == "__main__":
    main()
