"""
VERITAS-NLP: Türkçe ve İngilizce Veri Seti Birleştirme Aracı
=============================================================
Bu dosya:
1. Kaggle'dan Türkçe sahte haber veri setini indirir
2. Mevcut İngilizce WELFake verisiyle birleştirir
3. Birleşik veriyi 'combined_dataset.csv' olarak kaydeder

Çıktı: data/processed/combined_dataset.csv
"""

import os
import sys
import pandas as pd
import numpy as np
import re

def download_turkish_dataset():
    """Kaggle'dan Türkçe sahte haber veri setini indirir."""
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    # Olası dosya isimleri
    possible_files = [
        os.path.join(raw_dir, "turkish_fake_real.csv"),
        os.path.join(raw_dir, "clean.csv"),
        os.path.join(raw_dir, "turkish-fake-and-real-news", "clean.csv"),
    ]
    
    # Dosya zaten var mı kontrol et
    for f in possible_files:
        if os.path.exists(f):
            print(f"✅ Türkçe veri seti zaten mevcut: {f}")
            return f
    
    print("\n📥 Kaggle'dan Türkçe veri seti indiriliyor...")
    print("   Dataset: atakanak/turkish-fake-and-real-news")
    
    try:
        import opendatasets as od
        od.download("https://www.kaggle.com/datasets/atakanak/turkish-fake-and-real-news", 
                     data_dir=raw_dir)
        
        # İndirilen dosyayı bul
        download_dir = os.path.join(raw_dir, "turkish-fake-and-real-news")
        if os.path.exists(download_dir):
            for f in os.listdir(download_dir):
                if f.endswith('.csv'):
                    return os.path.join(download_dir, f)
    except Exception as e:
        print(f"⚠️ Otomatik indirme başarısız: {e}")
        print("\n📋 Manuel indirme talimatları:")
        print("   1. https://www.kaggle.com/datasets/atakanak/turkish-fake-and-real-news adresine gidin")
        print("   2. 'Download' butonuna tıklayın")
        print(f"   3. İndirilen CSV dosyasını '{raw_dir}' klasörüne koyun")
        print("   4. Dosya adını 'turkish_fake_real.csv' olarak değiştirin")
        print("   5. Bu scripti tekrar çalıştırın")
        sys.exit(1)
    
    return None


def clean_text_multilingual(text):
    """
    Çok dilli metin temizleme fonksiyonu.
    Hem İngilizce hem Türkçe karakterleri korur.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Küçük harf
    text = text.lower()
    
    # Sadece harfleri koru (İngilizce + Türkçe karakterler)
    # Türkçe özel karakterler: ç, ş, ğ, ü, ö, ı, İ
    text = re.sub(r'[^a-zçşğüöı\s]', ' ', text)
    
    # Fazla boşlukları temizle
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def load_and_prepare_turkish(file_path):
    """Türkçe veri setini yükler ve hazırlar."""
    print(f"\n📂 Türkçe veri seti yükleniyor: {file_path}")
    df = pd.read_csv(file_path)
    
    print(f"   Ham satır sayısı: {len(df)}")
    print(f"   Sütunlar: {df.columns.tolist()}")
    
    # Sütun isimlerini normalize et
    df.columns = [c.lower().strip() for c in df.columns]
    
    # 'content' veya 'text' sütununu bul
    content_col = None
    for col_name in ['content', 'text', 'clean_data', 'haber', 'metin', 'news']:
        if col_name in df.columns:
            content_col = col_name
            break
    
    if content_col is None:
        # İlk string sütunu dene
        for col in df.columns:
            if df[col].dtype == 'object' and col != 'label':
                content_col = col
                break
    
    if content_col is None:
        print("❌ HATA: Türkçe veri setinde metin sütunu bulunamadı!")
        print(f"   Mevcut sütunlar: {df.columns.tolist()}")
        sys.exit(1)
    
    # label sütununu bul
    label_col = None
    for col_name in ['label', 'labels', 'class', 'fake', 'etiket']:
        if col_name in df.columns:
            label_col = col_name
            break
    
    if label_col is None:
        print("❌ HATA: Türkçe veri setinde etiket sütunu bulunamadı!")
        sys.exit(1)
    
    # Yeni DataFrame oluştur
    result = pd.DataFrame({
        'content': df[content_col],
        'label': df[label_col]
    })
    
    # Temizle
    result = result.dropna(subset=['content', 'label'])
    result['content'] = result['content'].apply(clean_text_multilingual)
    result = result[result['content'].str.len() > 10]  # Çok kısa metinleri at
    
    # Label'ları kontrol et ve 0/1 formatına çevir
    unique_labels = result['label'].unique()
    print(f"   Benzersiz etiketler: {unique_labels}")
    
    # Eğer label 0/1 değilse dönüştür
    if set(unique_labels) != {0, 1}:
        label_map = {}
        for label in unique_labels:
            lbl_str = str(label).lower()
            if 'fake' in lbl_str or 'sahte' in lbl_str or 'yalan' in lbl_str:
                label_map[label] = 1
            elif 'real' in lbl_str or 'gerçek' in lbl_str or 'doğru' in lbl_str:
                label_map[label] = 0
            else:
                # Sayısal değer olabilir
                try:
                    label_map[label] = int(label)
                except:
                    pass
        
        if label_map:
            result['label'] = result['label'].map(label_map)
            result = result.dropna(subset=['label'])
            result['label'] = result['label'].astype(int)
    
    result['language'] = 'tr'
    
    print(f"   ✅ İşlenmiş Türkçe veri: {len(result)} satır")
    print(f"   📊 Dağılım: Real={len(result[result['label']==0])}, Fake={len(result[result['label']==1])}")
    
    return result


def load_english_data():
    """Mevcut İngilizce WELFake verisini yükler."""
    file_path = os.path.join("data", "processed", "WELFake_cleaned.csv")
    
    if not os.path.exists(file_path):
        print(f"❌ HATA: İngilizce veri seti bulunamadı: {file_path}")
        sys.exit(1)
    
    print(f"\n📂 İngilizce veri seti yükleniyor: {file_path}")
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['content', 'label'])
    
    # Sadece content ve label sütunlarını al
    result = df[['content', 'label']].copy()
    result['language'] = 'en'
    
    print(f"   ✅ İşlenmiş İngilizce veri: {len(result)} satır")
    print(f"   📊 Dağılım: Real={len(result[result['label']==0])}, Fake={len(result[result['label']==1])}")
    
    return result


def merge_and_save(en_df, tr_df):
    """İki veri setini birleştirir ve kaydeder."""
    print("\n🔀 Veri setleri birleştiriliyor...")
    
    combined = pd.concat([en_df, tr_df], ignore_index=True)
    
    # Karıştır (shuffle)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Kaydet
    output_path = os.path.join("data", "processed", "combined_dataset.csv")
    combined.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f" BİRLEŞTİRME TAMAMLANDI")
    print(f"{'='*60}")
    print(f"📁 Çıktı dosyası: {output_path}")
    print(f"📊 Toplam satır: {len(combined):,}")
    print(f"   🇬🇧 İngilizce: {len(combined[combined['language']=='en']):,}")
    print(f"   🇹🇷 Türkçe:    {len(combined[combined['language']=='tr']):,}")
    print(f"   ✅ Real (0):   {len(combined[combined['label']==0]):,}")
    print(f"   🔴 Fake (1):   {len(combined[combined['label']==1]):,}")
    
    return output_path


def main():
    print("=" * 60)
    print(" VERITAS-NLP: Çok Dilli Veri Seti Hazırlama")
    print("=" * 60)
    
    # 1. Türkçe veri setini indir/bul
    tr_file = download_turkish_dataset()
    
    if tr_file is None:
        print("❌ Türkçe veri seti bulunamadı!")
        return
    
    # 2. Verileri yükle
    tr_df = load_and_prepare_turkish(tr_file)
    en_df = load_english_data()
    
    # 3. Birleştir ve kaydet
    output_path = merge_and_save(en_df, tr_df)
    
    print(f"\n🎯 Sonraki adım: Modelleri yeniden eğitin:")
    print(f"   python scripts/train_bilstm.py")
    print(f"   python scripts/train_bert.py")


if __name__ == "__main__":
    main()
