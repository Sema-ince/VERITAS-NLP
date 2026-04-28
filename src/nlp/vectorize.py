import torch
from transformers import AutoTokenizer

# Proje için kullanılacak temel İngilizce BERT modeli (WELFake İngilizce olduğu için)
# Eğer Türkçeye dönülecekse: 'dbmdz/bert-base-turkish-cased' olarak değiştirilebilir.
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128  # Literatür taramasında belirlenen başlangıç parametrelerinden (128 veya 256)

class BertVectorizer:
    def __init__(self, model_name=MODEL_NAME, max_length=MAX_LENGTH):
        self.model_name = model_name
        self.max_length = max_length
        print(f"Tokenizer yükleniyor: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def vectorize(self, text):
        """
        Gelen metni (tekil string veya liste) BERT'in anlayacağı formata çevirir.
        Geriye input_ids ve attention_mask içeren bir sözlük (PyTorch tensor) döner.
        """
        if isinstance(text, str):
            text = [text] # Tek bir string ise listeye çevir

        encoded_inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return encoded_inputs

if __name__ == "__main__":
    # Kısa bir test
    vectorizer = BertVectorizer()
    sample_text = "This is a sample fake news text for vectorization test."
    output = vectorizer.vectorize(sample_text)
    print("Input IDs shape:", output['input_ids'].shape)
    print("Attention Mask shape:", output['attention_mask'].shape)
