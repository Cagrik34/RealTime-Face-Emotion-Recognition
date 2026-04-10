# Duygu Durumu Analizi Projesi

## 🎓 Proje Hakkında (About the Project)

Bu uygulama, İstanbul Beykent Üniversitesi **"Görüntü İşleme"** dersi bitirme projesi kapsamında geliştirilmiştir. Akademik bir çalışmanın ürünü olan bu projenin kaynak kodları, açık kaynak topluluğuna katkı sağlamak amacıyla **MIT Lisansı** ile özgür bırakılmıştır. 

Kod mimarisini inceleyebilir, yapay zeka modellerinden ilham alabilir veya projeyi kendi sistemlerinize entegre ederek dilediğiniz gibi geliştirebilirsiniz! 🚀

Bu proje, gerçek zamanlı yüz duygu analizi yapan modern bir Python uygulamasıdır.

## Gereksinimler

- Python 3.9.13
- Visual Studio Build Tools (dlib kurulumu için)

## Özellikler

- Gerçek zamanlı yüz tespiti (canlı kamera ile)
- **13 farklı duygu analizi:**
  - Çok Mutlu (0.50 eşik değeri)
  - Mutlu (0.35 eşik değeri)
  - Çok Üzgün (0.85 eşik değeri)
  - Üzgün (0.70 eşik değeri)
  - Çok Kızgın (0.85 eşik değeri)
  - Kızgın (0.70 eşik değeri)
  - Yoğun Korku (0.90 eşik değeri)
  - Korku (0.80 eşik değeri)
  - Çok İğrenme (0.85 eşik değeri)
  - İğrenme (0.70 eşik değeri)
  - Çok Şaşkın (0.80 eşik değeri)
  - Şaşkın (0.65 eşik değeri)
  - Nötr (0.30/0.15 eşik değerleri)
- Sadece **canlı analizde** maksimum 1 yüz üzerinde analiz (en büyük yüz)
- Resimden duygu analizi (tek tıkla resim seçip analiz)
- Duygu istatistikleri ve canlı grafikler (pasta ve çubuk)
- Veritabanında kayıt tutma (canlı ve resim analizleri ayrı kaydedilir)
- Modern ve kullanıcı dostu Tkinter arayüzü
- Modlar arası kolay geçiş: "Canlı Duygu Analizi" ve "Resimden Duygu Analizi" butonları

## Hızlı Başlangıç

1. Python 3.9.13 veya üstünü yükleyin
2. Visual Studio Build Tools'u yükleyin (dlib için gerekli)
3. Projeyi bilgisayarınıza indirin
4. Komut satırında proje klasörüne gidin
5. Gerekli kütüphaneleri yükleyin:
   ```
   pip install -r requirements.txt
   ```
6. Programı başlatın:
   ```
   python main.py
   ```

## Önemli Notlar

### Webcam Çakışması ve Canlı Analiz
Projenin canlı analiz özelliği, webcam kullanımı gerektirmektedir. Ancak, webcam kullanımı sırasında bazı uygulamalarla (örneğin, OBS Studio gibi) çakışma yaşanabilir. Bu durumda, webcam'e erişim sorunları oluşabilir. Bu nedenle, canlı analiz özelliğini kullanırken webcam'in başka bir uygulama tarafından kullanılmadığından emin olun.

Projenin sunumunda, canlı analiz görüntüsü ile webcam kaydı görüntüsü çakıştığından dolayı canlı analiz özelliğine kadar olan kısım detaylı bir şekilde anlatılmaktadır. Canlı analiz özelliği ise temel olarak videoda gösterilmektedir. Canlı analiz sırasında alınan örnek resimler, proje dosyasına eklenmiştir. Bu resimler, canlı analiz özelliğinin nasıl çalıştığını göstermektedir. Kullanıcılar, bu resimleri inceleyerek canlı analiz özelliğinin işleyişini daha iyi anlayabilirler.

- Tüm model dosyaları (`emotion_model.h5`, `shape_predictor_68_face_landmarks.dat`, `haarcascade_frontalface_default.xml`, `haarcascade_mcs_mouth.xml`) projenin ana dizininde olmalıdır.
- Veritabanı (`emotions.db`) otomatik oluşturulacaktır.
- Program ilk açıldığında modeller yüklenecek, bu birkaç saniye sürebilir.
- Canlı analiz için kameranızın çalışır durumda olması gerekir.
- İyi aydınlatma koşulları daha doğru sonuçlar verir.

## Dosya Yapısı

- `main.py`: Ana program ve arayüz
- `database.py`: Veritabanı işlemleri
- `emotion_detector.py`: Duygu tespiti ve mesh işlemleri
- `utils.py`: Yardımcı fonksiyonlar (grafik, kayıt vb.)
- `emotion_model.h5`: Eğitilmiş duygu analizi modeli
- `haarcascade_frontalface_default.xml`: Yüz tespiti için Haar Cascade
- `shape_predictor_68_face_landmarks.dat`: Yüz noktaları için dlib modeli

## Notlar

- Program çalışırken kameraya bakmanız gerekmektedir (canlı analiz için)
- İyi aydınlatma koşulları daha doğru sonuçlar verir
- Sadece canlı analizde maksimum 1 yüz analiz edilir
- Her duygu kategorisi için özel eşik değerleri kullanılmaktadır
- Program kapatıldığında otomatik grafik gösterimi yoktur, grafikler arayüzde anlık olarak güncellenir 

---

### Karakter Uyumluluğu Hakkında Not

> **Dikkat:** Kodun ve arayüzdeki tüm duygu isimleri ile kullanıcıya gösterilen mesajlarda Türkçe karakterler (ç, ğ, ı, ö, ş, ü, Ç, Ğ, İ, Ö, Ş, Ü) İngilizce karşılıklarına çevrilmiştir (örn. "Çok Mutlu" → "Cok Mutlu", "Üzgün" → "Uzgun").
>
> **Sebep:** Farklı işletim sistemlerinde, Python ortamlarında veya terminal/arayüzlerde Türkçe karakterler bozuk görünebilir veya hata verebilir. Teslimde ve farklı bilgisayarlarda sorunsuz çalışması için bu dönüşüm yapılmıştır. 
