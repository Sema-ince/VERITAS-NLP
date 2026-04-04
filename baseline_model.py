import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def run_real_baseline():
    """
    Ekip arkadaşımız Sinan tarafından temizlenip işlenmiş 
    gerçek WELFake veri setini kullanan NLP Baseline Modeli (2. Hafta Görevi)
    """
    
    # 1. Temizlenmiş veri yolunu belirle
    file_path = os.path.join("data", "processed", "WELFake_cleaned.csv")
    
    if not os.path.exists(file_path):
        print(f"HATA: Temizlenmiş veri seti bulunamadı -> {file_path}")
        print("Lütfen önce veriyi temizlemek için şu komutu çalıştırın: python scripts/preprocess_welfake.py")
        return

    # 2. Veriyi Yükleme
    print(f"🔄 Gerçek veriler yükleniyor ({file_path})...")
    df = pd.read_csv(file_path)
    
    # Güvenlik: Eksik verileri at
    df = df.dropna(subset=['content', 'label'])
    
    print("✅ Veri yüklendi! Toplam satır sayısı:", len(df))
    
    # 3. Veriyi Bölme (Eğitim ve Test)
    print("\n🔀 Veri seti bölünüyor (%80 Eğitim, %20 Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['content'], df['label'], test_size=0.2, random_state=42
    )
    
    # 4. TF-IDF ile Kelimeleri Rakamlara Çevirme
    print("🔢 TF-IDF kullanılarak metinler sayılara dönüştürülüyor...")
    # NOTE: 72k satır için RAM dolmaması adına max_features 10000 ile sınırlandı
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # 5. Logistic Regression Modeli Eğitimi
    print("\n🤖 1. Logistic Regression modeli eğitiliyor...")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_tfidf, y_train)
    lr_preds = lr_model.predict(X_test_tfidf)
    print(f"📊 Logistic Regression Doğruluğu: % {accuracy_score(y_test, lr_preds) * 100:.2f}")
    
    # 6. Random Forest Modeli Eğitimi
    print("\n🌲 2. Random Forest modeli eğitiliyor (Bu işlem biraz sürebilir)...")
    # n_jobs=-1 bilgisayarın tüm işlemci çekirdeklerini kullanır (Hızlandırır)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_tfidf, y_train)
    rf_preds = rf_model.predict(X_test_tfidf)
    
    print(f"📊 Random Forest Doğruluğu: % {accuracy_score(y_test, rf_preds) * 100:.2f}")
    print("\n📋 Random Forest Sınıflandırma Raporu:")
    print(classification_report(y_test, rf_preds, target_names=["Real (0)", "Fake (1)"], zero_division=0))

if __name__ == "__main__":
    run_real_baseline()
