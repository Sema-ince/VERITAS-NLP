"""
VERITAS-NLP: Ortak Ön İşleme Modülü (Unified Preprocessing)
=============================================================
Bu dosya, TÜM modüller tarafından kullanılan TEK metin temizleme 
fonksiyonunu içerir.

KURAL: Hem eğitim hem de uygulama (app.py) bu dosyadaki fonksiyonu 
       kullanmalıdır. Böylece eğitim-uygulama tutarsızlığı önlenir.
"""

import re


def preprocess_text(text, language='auto'):
    """
    Metni temizler: küçük harf, sadece harf karakterleri, fazla boşluk kaldırılır.
    
    Bu fonksiyon hem eğitimde hem app.py'de aynı şekilde kullanılmalıdır.
    Türkçe ve İngilizce karakterleri korur.
    
    Parametreler:
    - text: Temizlenecek metin
    - language: 'auto' (otomatik algılama), 'tr', veya 'en'
    """
    if not text:
        return ""
    text = str(text).lower()
    # Hem İngilizce hem Türkçe harfleri koru
    text = re.sub(r'[^a-zçşğüöı\s]', ' ', text)
    # Fazla boşlukları temizle
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def detect_language(text):
    """Metnin dilini basit bir sezgisel yöntemle algılar."""
    # Türkçeye özgü karakterler
    turkish_chars = set('çşğüöıÇŞĞÜÖİ')
    text_chars = set(text)
    if text_chars & turkish_chars:
        return 'tr'
    
    # Türkçeye özgü yaygın kelimeler
    turkish_words = {'bir', 've', 'ile', 'olan', 'için', 'bu', 'da', 'de',
                     'den', 'dan', 'olarak', 'ise', 'gibi', 'daha', 'sonra',
                     'kadar', 'gore', 'icin', 'ile', 'hem', 'ama', 'ancak',
                     'yarin', 'bugun', 'haber', 'haberi', 'acikladi', 'soyledi',
                     'turkiye', 'istanbul', 'ankara', 'devlet', 'bakani',
                     'cumhurbaskani', 'hukumet', 'basvurun', 'vatandaslarina',
                     'dagitacagini', 'buyudu', 'oraninda', 'kuruldu', 'ilinin',
                     'beylik', 'devleti', 'imparatorlugu', 'fethedip', 'vererek'}
    words = set(text.lower().split())
    turkish_match = len(words & turkish_words)
    if turkish_match >= 2:
        return 'tr'
    
    return 'en'
