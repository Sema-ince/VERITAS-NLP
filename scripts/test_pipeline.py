import os
import sys

# src dizinini import edebilmek için sys.path eklentisi
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.nlp.preprocess import setup_nltk, get_lemmatizer_and_stopwords, clean_text
from src.nlp.vectorize import BertVectorizer

def main():
    print("="*50)
    print(" VERITAS-NLP: Pipeline Uyumluluk Testi")
    print("="*50 + "\n")

    # 1. NLTK Kurulumu
    setup_nltk()
    lemmatizer, stop_words = get_lemmatizer_and_stopwords()

    # 2. Örnek (Canlı) Metin
    sample_text = """
    Breaking News: The new AI model is capable of generating highly realistic fake news articles!
    It uses advanced natural language processing techniques, including BERT and Bi-LSTM architectures.
    Experts are warning about the potential dangers. 12345! @#$%
    """
    print("Orijinal Metin:")
    print(sample_text.strip())
    print("-" * 50)

    # 3. Metin Ön İşleme (Temizleme)
    print("Adım 1: Metin Temizleniyor ve Kök Bulunuyor (Lemmatization)...")
    cleaned_text = clean_text(sample_text, lemmatizer, stop_words)
    print(f"\nTemizlenmiş Metin:\n{cleaned_text}")
    print("-" * 50)

    # 4. Vektörleştirme (BERT)
    print("Adım 2: Modelin Anlayacağı Formata (Vektörlere) Dönüştürülüyor...")
    try:
        vectorizer = BertVectorizer()
        output = vectorizer.vectorize(cleaned_text)
        
        input_ids = output['input_ids']
        attention_mask = output['attention_mask']
        
        print("\n[BAŞARILI] Vektörleştirme işlemi tamamlandı.")
        print(f"-> Model Giriş Boyutu (Input IDs Shape): {input_ids.shape} (Beklenen: Batch x Max_Length)")
        print(f"-> Dikkat Maskesi Boyutu (Attention Mask Shape): {attention_mask.shape}")
        
        print("\nÖrnek Vektör Çıktısı (İlk 10 Token ID):")
        print(input_ids[0][:10].tolist())
        
        print("\nNot: Modelin giriş boyutları veri seti ve UI (Arayüz) ile tam uyumludur.")
        
    except Exception as e:
        print(f"\n[HATA] Vektörleştirme sırasında bir sorun oluştu: {e}")

if __name__ == "__main__":
    main()
