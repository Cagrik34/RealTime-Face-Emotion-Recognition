import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import matplotlib
matplotlib.use('TkAgg')

def save_image(frame, emotion, confidence, intensity_score):
    """Tespit edilen duygu durumunu iceren goruntuyu kaydet"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"emotion_{emotion}_{timestamp}.jpg"
    
    # Görüntüyü kaydet
    cv2.imwrite(filename, frame)
    return filename

def plot_emotion_distribution(emotions):
    """Duygu dağılımını hem pasta hem bar grafik olarak görselleştirir"""
    if not emotions:
        print("Gosterilecek duygu verisi bulunamadi!")
        return

    # 7 temel duygu ve detayli duygular
    emotion_labels = [
        'Cok Mutlu', 'Mutlu',
        'Cok Uzgun', 'Uzgun',
        'Cok Kizgin', 'Kizgin',
        'Yogun Korku', 'Korku',
        'Cok Igrenme', 'Igrenme',
        'Cok Saskin', 'Saskin',
        'Notr'
    ]
    emotion_counts = {label: 0 for label in emotion_labels}
    for emotion in emotions:
        if emotion in emotion_counts:
            emotion_counts[emotion] += 1
        else:
            # Eğer yeni bir duygu varsa onu da ekle
            emotion_counts[emotion] = 1

    # Grafik
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plt.style.use('seaborn')

    # Pasta grafik (solda)
    axes[0].pie(
        [emotion_counts[label] for label in emotion_labels],
        labels=emotion_labels,
        autopct='%1.1f%%',
        startangle=140
    )
    axes[0].set_title('Duygu Dagilimi (Yuzde)')

    # Sütun grafik (sağda)
    bars = axes[1].bar(
        emotion_labels,
        [emotion_counts[label] for label in emotion_labels]
    )
    axes[1].set_title('Duygu Dagilimi (Sayi)')
    axes[1].set_xlabel('Duygular')
    axes[1].set_ylabel('Siklik')
    axes[1].set_xticklabels(emotion_labels, rotation=45, ha='right')
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(height)}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show(block=True)

def calculate_statistics(records):
    """Kayitlar uzerinden istatistikleri hesaplar"""
    if not records:
        print("Istatistik hesaplanamadi: Kayit bulunamadi!")
        return

    emotions = [record[1] for record in records]
    confidences = [record[2] for record in records]
    scores = [record[3] for record in records]

    print("\nIstatistikler:")
    print(f"Toplam Kayit: {len(records)}")
    print(f"Ortalama Guven: {sum(confidences) / len(confidences):.2f}")
    print(f"Ortalama Skor: {sum(scores) / len(scores):.2f}")
    print(f"En Yuksek Guven: {max(confidences):.2f}")
    print(f"En Dusuk Guven: {min(confidences):.2f}")
    print(f"En Yuksek Skor: {max(scores):.2f}")
    print(f"En Dusuk Skor: {min(scores):.2f}")

if __name__ == "__main__":
    # Test: Grafik fonksiyonu calisiyor mu?
    plot_emotion_distribution(['Mutlu', 'Uzgun', 'Mutlu', 'Kizgin', 'Mutlu', 'Uzgun']) 