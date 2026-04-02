import pandas as pd
import os

def main():
    # Veri setinin bilgisayardaki tam yolu (VERITAS-NLP/data/raw/WELFake_Dataset.csv)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "..", "data", "raw", "WELFake_Dataset.csv")
    
    print("\n" + "="*50)
    print(" VERITAS-NLP: WELFake Veri Seti Keşifsel Analizi")
    print("="*50 + "\n")
    
    if not os.path.exists(file_path):
        print(f"HATA: Veri seti bulunamadı -> {file_path}\n")
        print("LÜTFEN ŞU ADIMLARI İZLEYİN:")
        print("1. Kaggle'dan 'WELFake Dataset' adlı veri setini indirin.")
        print("   (Link: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification )")
        print("2. İndirdiğiniz arşivden çıkan 'WELFake_Dataset.csv' dosyasını")
        print("   'VERITAS-NLP/data/raw/' klasörünün içine kopyalayın.")
        print("3. Bu kodu tekrar çalıştırın.\n")
        return

    print(f"Veri seti yükleniyor: {os.path.basename(file_path)} ...\n")
    try:
        # Veriyi Pandas DataFrame olarak oku
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Dosya okunurken bir hata oluştu: {e}")
        return

    print("\n--- Veri Seti Genel Yapısı (Şekil) ---")
    print(f"Toplam Satır (Haber): {df.shape[0]:,}")
    print(f"Toplam Sütun (Özellik): {df.shape[1]}")
    
    print("\n--- Sütun İsimleri ve Veri Tipleri ---")
    print(df.dtypes)
    
    print("\n--- İlk 3 Satırın Önizlemesi ---")
    print(df.head(3))
    
    print("\n--- Eksik Veri (Null) Analizi ---")
    print(df.isnull().sum())
    
    print("\n--- Etiket (Label) Dağılımı ---")
    # WELFake veri setinde etiket genellikle 'label' sütununda yer alır (0 = Fake, 1 = Real vb. olabilir)
    if 'label' in df.columns:
        distribution = df['label'].value_counts()
        print(distribution)
        
        print("\nOransal Dağılım:")
        print((df['label'].value_counts(normalize=True) * 100).round(2).astype(str) + ' %')
    else:
        print("UYARI: 'label' adlı hedef sütun bulunamadı. Lütfen sütun isimlerini kontrol edin.")
        
    print("\n\nAnaliz başarıyla tamamlandı. Veri setinin yapısal incelemesi yapıldı.")
    print("Sonraki aşamada modelleme için NLP metin önişleme (cleaning, tokenization) adımlarına geçilebilir.")

if __name__ == "__main__":
    main()
