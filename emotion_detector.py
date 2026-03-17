import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model # type: ignore
import tensorflow as tf
from datetime import datetime
import time
import json
from utils import save_image, plot_emotion_distribution
from database import Database
import dlib # type: ignore

def turkce_to_ascii(text):
    replacements = {
        'ç': 'c', 'Ç': 'C',
        'ğ': 'g', 'Ğ': 'G',
        'ı': 'i', 'İ': 'I',
        'ö': 'o', 'Ö': 'O',
        'ş': 's', 'Ş': 'S',
        'ü': 'u', 'Ü': 'U',
        'â': 'a', 'Â': 'A',
        'î': 'i', 'Î': 'I',
        'û': 'u', 'Û': 'U'
    }
    for turkce, ascii in replacements.items():
        text = text.replace(turkce, ascii)
    return text

class EmotionDetector:
    def __init__(self):
        try:
            # Haar Cascade dosyalarinin yollarini belirle
            face_cascade_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'haarcascade_frontalface_default.xml')
            mouth_cascade_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'haarcascade_mcs_mouth.xml')
            print(f"Haar Cascade dosyalari yollari: {face_cascade_path}, {mouth_cascade_path}")
            
            # Haar Cascade siniflandiricilarini yukle
            self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
            self.mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)
            if self.face_cascade.empty() or self.mouth_cascade.empty():
                raise Exception("Haar Cascade dosyalari yuklenemedi!")
            print("Haar Cascade'ler basariyla yuklendi.")
            
            # Model dosyasinin yolunu belirle
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'emotion_model.h5')
            print(f"Model dosyasi yolu: {model_path}")
            
            # Duygu algilama modelini yukle
            self.emotion_model = load_model(model_path)
            print("Model basariyla yuklendi.")
            
            # Duygu etiketleri
            self.emotion_labels = ['Mutlu', 'Uzgun', 'Kizgin', 'Korku', 'Igrenme', 'Saskin', 'Notr']
            self.detailed_emotion_labels = []
            self.emotion_history = []
            self.confidence_history = []
            self.score_history = []
            self.previous_emotion = None
            self.last_frame = None
            
            # Veritabani baglantisi
            try:
                self.db = Database()
            except Exception as e:
                print(f"Veritabani baglanti hatasi: {str(e)}")
                self.db = None
            
            # dlib yuz landmark modelini yukle
            predictor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shape_predictor_68_face_landmarks.dat')
            if not os.path.exists(predictor_path):
                raise Exception("dlib landmark model dosyasi bulunamadi!")
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(predictor_path)
            print("dlib landmark modeli basariyla yuklendi.")
            
        except Exception as e:
            print(f"Baslatma hatasi: {str(e)}")
            raise

    def map_emotion_with_intensity(self, emotion, confidence):
        """
        Ana bir duyguyu, güven puanına göre daha detaylı, yoğunluk tabanlı bir etikete eşler.
        Her duygu için optimize edilmiş eşikler kullanılır.
        """
        # Her duygu için optimize edilmiş eşikler
        thresholds = {
            "Mutlu": {"çok": 0.50, "orta": 0.35},  # Mutlu için daha yüksek eşikler
            "Uzgun": {"çok": 0.85, "orta": 0.70},  # Üzgün için yüksek eşikler
            "Kizgin": {"çok": 0.85, "orta": 0.70},  # Kızgın için yüksek eşikler
            "Korku": {"çok": 0.90, "orta": 0.80},  # Korku için yüksek eşikler
            "Igrenme": {"çok": 0.85, "orta": 0.70},  # İğrenme için yüksek eşikler
            "Saskin": {"çok": 0.80, "orta": 0.65},  # Şaşkın için orta-yüksek eşikler
            "Notr": {"çok": 0.30, "orta": 0.15}  # Nötr için düşük eşikler
        }

        # Varsayılan nötr eşiği
        general_neutral_threshold = 0.15

        # Önce mutlu duygusunu kontrol et (öncelikli)
        if emotion == "Mutlu":
            if confidence >= thresholds["Mutlu"]["çok"]:
                return "Çok Mutlu"
            elif confidence >= thresholds["Mutlu"]["orta"]:
                return "Mutlu"
            else:
                return "Notr"
        
        # Sonra korku duygusunu kontrol et (daha yüksek eşikler)
        elif emotion == "Korku":
            if confidence >= thresholds["Korku"]["orta"]:
                return "Mutlu"
            else:
                return "Notr"
        
        # Diğer duygular için normal kontrol
        elif emotion in thresholds:
            if confidence >= thresholds[emotion]["çok"]:
                return f"Çok {emotion}"
            elif confidence >= thresholds[emotion]["orta"]:
                return emotion
            elif confidence < general_neutral_threshold:
                return "Notr"
            else:
                return emotion
        else:
            return "Notr"

    def detect_landmarks(self, frame, face_coords):
        """dlib ile yuz landmarklarini tespit eder"""
        x, y, w, h = face_coords
        # dlib icin yuz bolgesini belirle
        rect = dlib.rectangle(x, y, x + w, y + h)
        # Landmarkları tespit et
        shape = self.predictor(frame, rect)
        landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        return landmarks

    def detect_mouth(self, frame, face_coords):
        """Ağız bölgesini tespit eder"""
        x, y, w, h = face_coords
        # Ağız genellikle yüzün alt yarısında bulunur
        roi = frame[y + h//2:y + h, x:x + w]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mouths = self.mouth_cascade.detectMultiScale(gray_roi, 1.1, 4)
        
        if len(mouths) > 0:
            # En büyük ağız bölgesini al
            mouth = max(mouths, key=lambda m: m[2] * m[3])
            mx, my, mw, mh = mouth
            # Orijinal koordinatlara çevir
            return (x + mx, y + h//2 + my, mw, mh)
        return None

    def analyze_landmarks(self, landmarks, mouth_coords=None):
        """Landmark noktalarindan ve ağız tespitinden ek analizler yapar"""
        # Agiz acikligi (noktalar 48-68 arasi)
        mouth_points = landmarks[48:68]
        mouth_height = max(p[1] for p in mouth_points) - min(p[1] for p in mouth_points)
        mouth_width = max(p[0] for p in mouth_points) - min(p[0] for p in mouth_points)
        mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0

        # Kas arasi mesafe (noktalar 17-27 arasi)
        eyebrow_points = landmarks[17:27]
        eyebrow_distance = max(p[0] for p in eyebrow_points) - min(p[0] for p in eyebrow_points)

        # Goz acikligi (noktalar 36-48 arasi)
        eye_points = landmarks[36:48]
        eye_height = max(p[1] for p in eye_points) - min(p[1] for p in eye_points)
        eye_width = max(p[0] for p in eye_points) - min(p[0] for p in eye_points)
        eye_ratio = eye_height / eye_width if eye_width > 0 else 0

        # Ağız tespiti sonuçlarını ekle
        mouth_detection = False
        if mouth_coords:
            mouth_detection = True

        return {
            'mouth_ratio': mouth_ratio,
            'eyebrow_distance': eyebrow_distance,
            'eye_ratio': eye_ratio,
            'mouth_detection': mouth_detection
        }

    def save_emotion(self, emotion, confidence, score):
        """Duygu kaydını veritabanına ekler"""
        if self.db:
            self.db.add_record(emotion, confidence, score)

    def detect_emotion(self, frame):
        """
        Verilen karedeki yüzleri ve duygularını tespit eder.
        Her duygu için optimize edilmiş güven artırımı uygulanır.
        """
        self.last_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
        results = []
        for (x, y, w, h) in faces:
            try:
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (48, 48))
                face_roi = face_roi.astype("float") / 255.0
                face_roi = np.expand_dims(face_roi, axis=0)
                face_roi = np.expand_dims(face_roi, axis=-1)
                preds = self.emotion_model.predict(face_roi, verbose=0)[0]
                if not preds.size:
                    print("Yüz ROI için tahmin üretilmedi, atlanıyor.")
                    continue
                emotion_idx = np.argmax(preds)
                emotion = self.emotion_labels[emotion_idx]
                confidence = float(preds[emotion_idx])
                score = float(np.max(preds))
                landmarks = self.detect_landmarks(frame, (x, y, w, h))
                mouth_coords = self.detect_mouth(frame, (x, y, w, h))
                landmark_analysis = self.analyze_landmarks(landmarks, mouth_coords)
                # Duyguya özel güven artırımı
                if landmark_analysis['mouth_detection']:
                    if emotion == "Mutlu":
                        confidence = min(1.0, confidence * 1.20)  # Mutlu için %20 artış
                    elif emotion == "Notr":
                        confidence = min(1.0, confidence * 1.15)  # Nötr için %15 artış
                    elif emotion == "Saskin":
                        confidence = min(1.0, confidence * 1.15)  # Şaşkın için %15 artış
                    elif emotion == "Igrenme":
                        confidence = min(1.0, confidence * 1.10)  # İğrenme için %10 artış
                    else:
                        confidence = min(1.0, confidence * 1.05)  # Diğer duygular için %5 artış
                # Duygu eşiklerini kontrol ederek nihai detaylı duyguyu belirle
                detailed_emotion = self.map_emotion_with_intensity(emotion, confidence)
                # Duygu stabilizasyonu
                if self.previous_emotion is not None and self.previous_emotion == emotion:
                    stable_emotion = emotion
                    stable_detailed = detailed_emotion
                else:
                    stable_emotion = emotion
                    stable_detailed = detailed_emotion
                    self.previous_emotion = emotion
                self.emotion_history.append(stable_emotion)
                self.detailed_emotion_labels.append(stable_detailed)
                self.confidence_history.append(confidence)
                self.score_history.append(score)
                results.append({
                    'bbox': (x, y, w, h),
                    'emotion': stable_emotion,
                    'detailed': stable_detailed,
                    'confidence': confidence,
                    'landmarks': landmarks,
                    'landmark_analysis': landmark_analysis,
                    'mouth_coords': mouth_coords
                })
            except Exception as e:
                print(f"({x}, {y}, {w}, {h}) konumundaki yüz işlenirken hata oluştu: {e}")
                continue
        return results

    def draw_results(self, frame, emotion, confidence, score, face_coords, landmarks=None, mouth_coords=None):
        """Sonuclari goruntu uzerine cizer ve landmark noktalarini gosterir"""
        if face_coords is None:
            return frame
        x, y, w, h = face_coords
        # Yüz çerçevesi (yeşil)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        text = f"{emotion} ({confidence:.2f})"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        score_text = f"Skor: {score:.2f}"
        cv2.putText(frame, score_text, (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Landmark noktalarını çiz (beyaz)
        if landmarks:
            for px, py in landmarks:
                cv2.circle(frame, (px, py), 2, (255, 255, 255), -1)
        
        # Ağız bölgesini çiz (kırmızı)
        if mouth_coords:
            mx, my, mw, mh = mouth_coords
            cv2.rectangle(frame, (mx, my), (mx+mw, my+mh), (0, 0, 255), 2)
        
        return frame

    def save_results(self):
        """Sonuclari kaydeder"""
        if not self.emotion_history:
            return
            
        # Sonuclari JSON olarak kaydet
        results = {
            'timestamp': datetime.now().isoformat(),
            'emotions': self.detailed_emotion_labels,
            'confidences': self.confidence_history,
            'scores': self.score_history
        }
        
        # JSON dosyasına kaydet
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        # Veritabanına kaydet
        if self.db:
            for emotion, confidence, score in zip(self.detailed_emotion_labels, self.confidence_history, self.score_history):
                if confidence >= 0.5:
                    self.db.save_emotion(emotion, confidence, score)
        
        # Görüntüyü kaydet
        if self.last_frame is not None:
            save_image(self.last_frame, 'last_frame.jpg')
        
        # Duygu dağılımını çiz
        plot_emotion_distribution(self.detailed_emotion_labels)
        
        print("Sonuclar kaydedildi.")
        
    def cleanup(self):
        """Kaynakları temizler"""
        if self.db:
            self.db.close()
        self.face_cascade = None
        self.mouth_cascade = None
        self.emotion_model = None
        self.detector = None
        self.predictor = None 