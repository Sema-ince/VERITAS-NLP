import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def setup_nltk():
    """NLTK kütüphanesi için gerekli veri dosyalarını indirir."""
    print("NLTK veri dosyaları kontrol ediliyor/indiriliyor...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print("NLTK dosyaları hazır.\n")

def get_lemmatizer_and_stopwords():
    """Lemmatizer ve Stopwords setini döndürür."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    return lemmatizer, stop_words

def clean_text(text, lemmatizer, stop_words):
    """
    Metin NLP ön işleme adımları (Lemmatization ile):
    1. Lowercase (Küçük harfe çevirme)
    2. Noktalama, özel karakterler ve sayıları kaldırma (Sadece harfler kalsın)
    3. Tokenization (Kelime kelime ayırma)
    4. Stopwords kaldırma
    5. Lemmatize (Kök/Gövde bulma)
    """
    if pd.isna(text):
        return ""
    
    # 1. Küçük harf
    text = str(text).lower()
    
    # 2. Noktalama, rakam ve özel karakterleri atma (Sadece a-z arası kalır)
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # 3. Fazla boşlukları temizleme
    text = re.sub(r'\s+', ' ', text).strip()
    
    if not text:
        return ""
    
    # Tokenization & Stopwords & Lemmatization aynı döngüde performans için
    tokens = text.split()
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return ' '.join(cleaned_tokens)
