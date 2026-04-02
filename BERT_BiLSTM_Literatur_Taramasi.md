# 📄 BERT ve Bi-LSTM Mimarileri Üzerine Literatür Taraması (1. Hafta)

## 📌 Proje Kapsamı

Bu hafta VERITAS-NLP projesi kapsamında, sahte haber tespiti için kullanılacak modeller üzerine bir ön araştırma yaptık. Özellikle BERT ve Bi-LSTM mimarilerine odaklanarak hem literatürdeki yerlerini inceledik hem de model için başlangıç parametrelerini belirlemeye çalıştık.

---

## 📖 1. Literatür Taraması

Doğal Dil İşleme alanında, özellikle sahte haber tespiti gibi konularda metni doğru anlamak oldukça önemli. Bu noktada çift yönlü bağlamı yakalayabilen modeller öne çıkıyor. BERT ve Bi-LSTM da aslında tam olarak bu ihtiyacı karşılayan iki güçlü yaklaşım.

### 1.1 BERT

BERT, Google tarafından geliştirilmiş bir model ve en önemli özelliği metni hem sağdan hem soldan okuyarak anlam çıkarması. Yani kelimeleri tek başına değil, bağlam içinde değerlendiriyor.

Sahte haber tespitinde bu oldukça işe yarıyor çünkü bu tür metinlerde genelde ince anlam oyunları oluyor. Literatürde de BERT’in klasik yöntemlere göre çok daha başarılı olduğu görülüyor. Özellikle Türkçe için hazır modeller kullanıldığında sonuçlar oldukça iyi.

---

### 1.2 Bi-LSTM

Bi-LSTM ise daha çok metnin sıralı yapısını anlamaya odaklanıyor. Metni hem baştan sona hem de sondan başa okuyarak kelimeler arasındaki uzun mesafeli ilişkileri yakalayabiliyor.

Haber metinleri genelde uzun olduğu için bu yapı avantaj sağlıyor. Yani bir kelimenin anlamını, metnin başka bir yerindeki bilgiyle ilişkilendirebiliyor.

---

### 1.3 BERT + Bi-LSTM (Hibrit Yaklaşım)

Son çalışmalar gösteriyor ki bu iki modeli birlikte kullanmak oldukça mantıklı. BERT metnin anlamını güçlü bir şekilde çıkarıyor, Bi-LSTM ise bu anlamı sıralı yapı içinde işliyor.

Bu yüzden hibrit bir model kurmak, özellikle sahte haber gibi karmaşık problemler için daha iyi sonuç verebilir gibi görünüyor.

---

## ⚙️ 2. Taslak Model Parametreleri

Literatüre bakarak ve genel uygulamaları inceleyerek başlangıç için bazı parametreler belirledik.

### BERT için

* Model: `bert-base-turkish-cased`
* Max length: 128 veya 256
* Dropout: 0.3

### Bi-LSTM için

* Hidden size: 128 veya 256
* Katman sayısı: 1 veya 2
* Bidirectional: aktif

### Eğitim ayarları

* Epoch: 3–5
* Batch size: 16 veya 32
* Learning rate: 2e-5 civarı
* Optimizer: AdamW
* Loss: Cross-Entropy

---

## 🎯 Sonuç

Bu hafta yaptığımız çalışma sonucunda, model için en uygun yaklaşımın BERT ile başlamak ve gerekirse Bi-LSTM ile desteklemek olduğu sonucuna vardık.

Ayrıca eğitim sürecinde kullanılacak parametreler için de bir başlangıç noktası belirledik. İlerleyen haftalarda bu değerleri veri setine göre daha iyi hale getirmeyi planlıyoruz.

---