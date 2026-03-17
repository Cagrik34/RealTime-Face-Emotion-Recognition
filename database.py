import sqlite3
from datetime import datetime
import os

os.environ["MEDIAPIPE_CACHE_DIR"] = "D:/temp"

class Database:
    def __init__(self):
        self.db_path = 'D:/duygudurumu/emotions.db'
        self.conn = None
        self.cursor = None
        self.connect()
        self.create_tables()

    def connect(self):
        """Veritabanına bağlanır"""
        try:
            # Veritabanı dizininin varlığını kontrol et
            db_dir = os.path.dirname(self.db_path)
            if not os.path.exists(db_dir):
                os.makedirs(db_dir)
                print(f"Veritabanı dizini oluşturuldu: {db_dir}")

            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            print(f"Veritabani dosyasi yolu: {self.db_path}")
            print("Veritabanı başarıyla bağlandı.")
        except sqlite3.Error as e:
            print(f"Veritabanı bağlantı hatası: {e}")
            raise
    
    def create_tables(self):
        """Gerekli tabloları oluşturur"""
        try:
            # Emotions tablosunu oluştur
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS emotions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    emotion TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    score REAL NOT NULL,
                    source TEXT NOT NULL,
                    image_path TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.conn.commit()
            print("Tablolar başarıyla oluşturuldu.")
        except sqlite3.Error as e:
            print(f"Tablo oluşturma hatası: {e}")
            raise
    
    def save_emotion(self, emotion, confidence, score, source='live', image_path=None):
        """Duygu verisini veritabanına kaydeder"""
        try:
            self.cursor.execute('''
                INSERT INTO emotions (emotion, confidence, score, source, image_path)
                VALUES (?, ?, ?, ?, ?)
            ''', (emotion, confidence, score, source, image_path))
            self.conn.commit()
            print(f"Kayıt ekleniyor: Duygu={emotion}, Güven={confidence:.2f}, Skor={score:.2f}, Kaynak={source}")
        except sqlite3.Error as e:
            print(f"Veri kaydetme hatası: {e}")
            self.conn.rollback()
    
    def get_all_emotions(self):
        """Tüm duygu verilerini getirir"""
        try:
            self.cursor.execute('SELECT * FROM emotions ORDER BY timestamp DESC')
            records = self.cursor.fetchall()
            print(f"Toplam {len(records)} kayıt bulundu.")
            return records
        except Exception as e:
            print(f"Veri okuma hatası: {str(e)}")
            return []
    
    def get_emotions_by_date(self, start_date, end_date):
        """Belirli tarih aralığındaki duygu verilerini getirir"""
        try:
            self.cursor.execute('''
            SELECT * FROM emotions 
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp DESC
            ''', (start_date, end_date))
            return self.cursor.fetchall()
        except Exception as e:
            print(f"Veri okuma hatası: {str(e)}")
            return []
    
    def get_emotion_stats(self):
        """Duygu istatistiklerini getirir"""
        try:
            self.cursor.execute('''
            SELECT 
                emotion,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence,
                AVG(score) as avg_score
            FROM emotions
            GROUP BY emotion
            ORDER BY count DESC
            ''')
            return self.cursor.fetchall()
        except Exception as e:
            print(f"İstatistik hesaplama hatası: {str(e)}")
            return []
    
    def get_recent_emotions(self, limit=10):
        """Son duygu verilerini getirir"""
        try:
            self.cursor.execute('''
            SELECT * FROM emotions 
            ORDER BY timestamp DESC 
            LIMIT ?
            ''', (limit,))
            return self.cursor.fetchall()
        except Exception as e:
            print(f"Veri okuma hatası: {str(e)}")
            return []
    
    def get_session_records(self):
        """Tüm oturum kayıtlarını getirir"""
        try:
            self.cursor.execute('''
                SELECT timestamp, emotion, confidence, score 
                FROM emotions 
                ORDER BY timestamp DESC
            ''')
            records = self.cursor.fetchall()
            print(f"Oturumda {len(records)} kayıt bulundu.")
            return records
        except Exception as e:
            print(f"Kayıtlar alınırken hata: {str(e)}")
            return []
    
    def get_live_records(self):
        """Canlı analiz kayıtlarını getirir"""
        try:
            self.cursor.execute('''
                SELECT * FROM emotions 
                WHERE source = 'live'
                ORDER BY timestamp DESC
            ''')
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Kayıt alma hatası: {e}")
            return []
    
    def get_image_records(self):
        """Resim analizi kayıtlarını getirir"""
        try:
            self.cursor.execute('''
                SELECT * FROM emotions 
                WHERE source = 'image'
                ORDER BY timestamp DESC
            ''')
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Kayıt alma hatası: {e}")
            return []
    
    def show_source_records(self, source, limit=5):
        """Belirli bir kaynaktan son N kaydı gösterir"""
        try:
            self.cursor.execute('''
                SELECT emotion, confidence, score, timestamp 
                FROM emotions 
                WHERE source = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (source, limit))
            records = self.cursor.fetchall()
            
            if records:
                print(f"\nSon {len(records)} {source} kaydı:")
                for record in records:
                    print(f"Duygu: {record[0]}, Güven: {record[1]:.2f}, Skor: {record[2]:.2f}, Zaman: {record[3]}")
            else:
                print(f"\n{source} kaynağında kayıt bulunamadı.")
        except sqlite3.Error as e:
            print(f"Kayıt gösterme hatası: {e}")
    
    def show_last_records(self, limit=5):
        """Son N kaydı yazdırır"""
        try:
            self.cursor.execute('''
                SELECT timestamp, emotion, confidence, score
                FROM emotions
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            records = self.cursor.fetchall()
            
            if not records:
                print("Kayıt bulunamadı!")
                return
            
            print(f"\nSon {limit} Kayıt:")
            print("-" * 50)
            for record in records:
                print(f"Zaman    : {record[0]}")
                print(f"Duygu    : {record[1]}")
                print(f"Güven    : {record[2]:.2f}")
                print(f"Skor     : {record[3]:.2f}")
                print("-" * 50)
        except Exception as e:
            print(f"Kayıt gösterme hatası: {str(e)}")
    
    def clear_database(self):
        """Veritabanındaki tüm kayıtları temizler"""
        try:
            self.cursor.execute('DELETE FROM emotions')
            self.conn.commit()
            print("Veritabanı başarıyla temizlendi.")
            return True
        except sqlite3.Error as e:
            print(f"Veritabanı temizleme hatası: {e}")
            self.conn.rollback()
            return False
    
    def __del__(self):
        """Veritabanı bağlantısını kapatır"""
        if self.conn:
            self.conn.close()
            print("Veritabanı bağlantısı kapatıldı.") 