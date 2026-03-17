import cv2
import numpy as np
from emotion_detector import EmotionDetector
import time
from datetime import datetime
import os
from utils import plot_emotion_distribution, calculate_statistics
from database import Database
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import sqlite3
import sys
matplotlib.use('TkAgg')

os.environ["TEMP"] = "D:/temp"
os.environ["TMP"] = "D:/temp"
os.environ["MEDIAPIPE_CACHE_DIR"] = "D:/temp"

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

class ConsoleOutputRedirector:
    def __init__(self, original_stdout):
        self.original_stdout = original_stdout
        self.output_buffer = []

    def write(self, text):
        self.original_stdout.write(text)
        self.output_buffer.append(text)

    def flush(self):
        self.original_stdout.flush()

    def get_output(self):
        return "".join(self.output_buffer)

class EmotionAnalysisApp:
    def __init__(self):
        # Konsol çıktısını yönlendir
        self.console_redirector = ConsoleOutputRedirector(sys.stdout)
        sys.stdout = self.console_redirector

        print("\nUygulama başlatılıyor...")
        self.root = tk.Tk()
        self.root.title(turkce_to_ascii("Duygu Analizi"))
        self.root.geometry("1200x800")
        
        # Tema değişkeni
        self.is_dark_mode = False
        
        # Canlı analiz için değişkenler
        self.cap = None
        self.is_live_running = False
        self.prev_time = None
        self.fps_list = []
        self.last_emotion = None
        
        # Ana frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Sol frame (butonlar ve görüntü için)
        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Butonlar için frame
        self.button_frame = tk.Frame(self.left_frame)
        self.button_frame.pack(pady=10)
        
        # Tema değiştirme butonu
        self.theme_button = tk.Button(self.button_frame, text=turkce_to_ascii("🌙 Karanlık Mod"), 
                                    command=self.toggle_theme,
                                    bg="#2196F3", fg="white", font=("Arial", 12))
        self.theme_button.pack(side=tk.LEFT, padx=5)
        
        # Butonlar
        self.live_button = tk.Button(self.button_frame, text=turkce_to_ascii("Canlı Duygu Analizi"), 
                                   command=self.start_live_analysis,
                                   bg="#4CAF50", fg="white", font=("Arial", 12))
        self.live_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = tk.Button(self.button_frame, text=turkce_to_ascii("Canlı Analizi Durdur"), 
                                   command=self.stop_live_analysis,
                                   bg="#f44336", fg="white", font=("Arial", 12),
                                   state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.image_button = tk.Button(self.button_frame, text=turkce_to_ascii("Resimden Duygu Analizi"), 
                                    command=self.analyze_image,
                                    bg="#2196F3", fg="white", font=("Arial", 12))
        self.image_button.pack(side=tk.LEFT, padx=5)
        
        # İstatistik butonu
        self.stats_button = tk.Button(
            self.button_frame,
            text="İstatistikleri Göster",
            command=self.show_statistics,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12, "bold"),
            width=20,
            height=2
        )
        self.stats_button.pack(pady=10)

        # Veritabanı temizleme butonu
        self.clear_db_button = tk.Button(
            self.button_frame,
            text="Veritabanını Temizle",
            command=self.clear_database,
            bg="#f44336",  # Kırmızı renk
            fg="white",
            font=("Arial", 12, "bold"),
            width=20,
            height=2
        )
        self.clear_db_button.pack(pady=10)
        
        # Görüntü için frame
        self.image_frame = ttk.Frame(self.left_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Görüntü etiketi
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Sonuç metni
        self.result_text = ttk.Label(self.left_frame, text="")
        self.result_text.pack(pady=5)
        
        # Sağ frame (istatistikler için)
        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # İstatistik tablosu
        self.stats_frame = ttk.Frame(self.right_frame)
        self.stats_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.tree = ttk.Treeview(self.stats_frame, columns=("Duygu", "Sayı", "Oran"), show="headings", height=10)
        self.tree.heading("Duygu", text=turkce_to_ascii("Duygu"))
        self.tree.heading("Sayı", text=turkce_to_ascii("Sayı"))
        self.tree.heading("Oran", text=turkce_to_ascii("Oran"))
        self.tree.column("Duygu", width=100)
        self.tree.column("Sayı", width=100)
        self.tree.column("Oran", width=100)
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Grafikler için frame
        self.graph_frame = ttk.Frame(self.right_frame)
        self.graph_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Grafikler
        self.fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.ax1 = self.fig.add_subplot(211)  # Pasta grafik
        self.ax2 = self.fig.add_subplot(212)  # Çubuk grafik
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Model ve veritabanı başlatma
        self.detector = EmotionDetector()
        self.db = Database()
        
        # İlk istatistikleri göster
        self.update_statistics()
        
        # Stil ayarları
        self.style = ttk.Style()
        self.apply_theme()

        # Veritabanı bağlantısı
        self.db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emotions.db")
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
        # Tabloları oluştur
        self.create_tables()
        
        # Veritabanı durumu etiketi
        self.db_status_label = tk.Label(self.root, text="")
        self.db_status_label.pack(pady=5)
        
        # Veritabanı durumunu güncelle
        self.update_db_status()

        # Uygulama kapanırken çalışacak protokolü ayarla
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def toggle_theme(self):
        """Tema değiştirme fonksiyonu"""
        self.is_dark_mode = not self.is_dark_mode
        self.apply_theme()
        self.theme_button.config(text=turkce_to_ascii("☀️ Aydınlık Mod") if self.is_dark_mode else turkce_to_ascii("🌙 Karanlık Mod"))

    def apply_theme(self):
        """Tema ayarlarını uygular"""
        if self.is_dark_mode:
            # Karanlık tema renkleri
            bg_color = "#2b2b2b"
            fg_color = "#ffffff"
            button_bg = "#404040"
            button_fg = "#ffffff"
            tree_bg = "#404040"
            tree_fg = "#ffffff"
            graph_bg = "#2b2b2b"
            graph_fg = "#ffffff"
            heading_bg = "#1c1c1c"  # Çok daha koyu bir gri tonu
            heading_fg = "#ffffff"  # Beyaz yazı
        else:
            # Aydınlık tema renkleri
            bg_color = "#ffffff"
            fg_color = "#000000"
            button_bg = "#f0f0f0"
            button_fg = "#000000"
            tree_bg = "#ffffff"
            tree_fg = "#000000"
            graph_bg = "#ffffff"
            graph_fg = "#000000"
            heading_bg = "#e0e0e0"  # Daha belirgin açık arka plan
            heading_fg = "#000000"  # Siyah yazı

        # Ana pencere ve frame'ler
        self.root.configure(bg=bg_color)
        self.main_frame.configure(style='Main.TFrame')
        self.left_frame.configure(style='Main.TFrame')
        self.right_frame.configure(style='Main.TFrame')
        self.image_frame.configure(style='Main.TFrame')
        self.stats_frame.configure(style='Main.TFrame')
        self.graph_frame.configure(style='Main.TFrame')

        # Stil ayarları
        self.style.configure('Main.TFrame', background=bg_color)
        self.style.configure('TLabel', background=bg_color, foreground=fg_color)
        self.style.configure('Treeview', background=tree_bg, foreground=tree_fg, fieldbackground=tree_bg)
        
        # Tablo başlık stili
        self.style.configure('Treeview.Heading', 
                           background=heading_bg, 
                           foreground=heading_fg,
                           relief='flat',
                           borderwidth=0,
                           font=('Arial', 10, 'bold')) 
        
        # Tablo içeriği stili
        self.style.map('Treeview',
                      background=[('selected', '#0078D7')],
                      foreground=[('selected', '#ffffff')])

        # Grafik tema ayarları
        plt.style.use('dark_background' if self.is_dark_mode else 'default')
        self.fig.set_facecolor(graph_bg)
        self.ax1.set_facecolor(graph_bg)
        self.ax2.set_facecolor(graph_bg)
        self.ax1.tick_params(colors=graph_fg)
        self.ax2.tick_params(colors=graph_fg)
        self.ax1.xaxis.label.set_color(graph_fg)
        self.ax1.yaxis.label.set_color(graph_fg)
        self.ax2.xaxis.label.set_color(graph_fg)
        self.ax2.yaxis.label.set_color(graph_fg)
        self.ax1.title.set_color(graph_fg)
        self.ax2.title.set_color(graph_fg)

        # Buton renkleri
        button_colors = {
            self.live_button: "#4CAF50",
            self.stop_button: "#f44336",
            self.image_button: "#2196F3",
            self.theme_button: "#2196F3"
        }
        
        for button, color in button_colors.items():
            button.configure(bg=color, fg="white")

        # Grafikleri güncelle
        self.update_statistics()

    def update_statistics(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.ax1.clear()
        self.ax2.clear()
        emotion_counts = {}
        records = self.detector.db.get_all_emotions() # Kayıtları al
        total = len(records)
        if total == 0:
            self.tree.insert("", tk.END, values=("-", 0, "0.0%"))
            self.ax1.text(0.5, 0.5, 'Veri yok', ha='center', va='center', fontsize=14)
            self.ax2.text(0.5, 0.5, 'Veri yok', ha='center', va='center', fontsize=14)
            self.fig.tight_layout()
            self.canvas.draw()
            print("Güncellenecek istatistik bulunamadı.")
            return
        for record in records:
            emotion = turkce_to_ascii(record[1])
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        for emotion, count in emotion_counts.items():
            percentage = (count / total) * 100 if total > 0 else 0
            if percentage != percentage:  # NaN kontrolü
                percentage = 0
            self.tree.insert("", tk.END, values=(emotion, count, f"{percentage:.1f}%"))
        # İstatistikleri konsola yazdır
        if records:
            from utils import calculate_statistics
            calculate_statistics(records)
        if emotion_counts:
            self.ax1.pie(emotion_counts.values(), labels=emotion_counts.keys(), autopct='%1.1f%%')
            self.ax1.set_title(turkce_to_ascii('Duygu Dagilimi (Pasta Grafigi)'))
            emotions_labels = list(emotion_counts.keys())
            counts = list(emotion_counts.values())
            self.ax2.bar(emotions_labels, counts)
            self.ax2.set_title(turkce_to_ascii('Duygu Dagilimi (Sutun Grafigi)'))
            self.ax2.set_ylabel(turkce_to_ascii('Sayi'))
            plt.setp(self.ax2.xaxis.get_majorticklabels(), rotation=45)
        self.fig.tight_layout()
        self.canvas.draw()

    def show_frame(self):
        if not self.is_live_running:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.is_live_running = False
            self.cap.release()
            return
        curr_time = time.time()
        if self.prev_time is None or self.prev_time == 0:
            fps = 0
        else:
            fps = 1 / (curr_time - self.prev_time)
        self.prev_time = curr_time
        self.fps_list.append(fps)
        if len(self.fps_list) > 30:
            self.fps_list.pop(0)
        avg_fps = sum(self.fps_list) / len(self.fps_list)
        results = self.detector.detect_emotion(frame)
        info_text = ""
        if results:
            result = results[0]  # Sadece ilk yüzü analiz et
            bbox = result['bbox']
            emotion = turkce_to_ascii(result['emotion'])
            confidence = result['confidence']
            # Mutlu ve Korku duygularının yerini değiştir
            if emotion == "Mutlu":
                if confidence >= 0.25:
                    emotion = "Yogun Korku"
                elif confidence >= 0.15:
                    emotion = "Korku"
                else:
                    emotion = "Notr"
            elif emotion == "Uzgun":
                if confidence > 0.85:
                    emotion = "Çok Uzgun"
                elif confidence > 0.70:
                    emotion = "Uzgun"
                else:
                    emotion = "Notr"
            elif emotion == "Kizgin":
                if confidence > 0.85:
                    emotion = "Çok Kizgin"
                elif confidence > 0.70:
                    emotion = "Kizgin"
                else:
                    emotion = "Notr"
            elif emotion == "Korku":
                if confidence > 0.85:
                    emotion = "Çok Mutlu"
                elif confidence > 0.70:
                    emotion = "Mutlu"
                else:
                    emotion = "Notr"
            elif emotion == "Igrenme":
                if confidence > 0.85:
                    emotion = "Çok Igrenme"
                elif confidence > 0.70:
                    emotion = "Igrenme"
                else:
                    emotion = "Notr"
            elif emotion == "Saskin":
                if confidence > 0.80:
                    emotion = "Çok Saskin"
                elif confidence > 0.65:
                    emotion = "Saskin"
                else:
                    emotion = "Notr"
            # Düşük güvenli tespitleri nötr olarak işaretle
            elif confidence < 0.4:
                emotion = "Notr"
            landmarks = result['landmarks']
            analysis = result['landmark_analysis']
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f"{emotion} ({confidence:.2f})"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            for px, py in landmarks:
                cv2.circle(frame, (px, py), 1, (255, 255, 255), -1)
            analysis_text = f"Agiz: {analysis['mouth_ratio']:.2f}, Kas: {analysis['eyebrow_distance']:.2f}, Goz: {analysis['eye_ratio']:.2f}"
            cv2.putText(frame, analysis_text, (x, y+h+50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            self.db.save_emotion(emotion, confidence, confidence, source='live')
            info_text = f"Duygu: {emotion}\nGuven: {confidence:.2f}\nFPS: {avg_fps:.2f}"
            self.last_emotion = emotion
        cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((640, 480))
        imgtk = ImageTk.PhotoImage(image=img)
        self.image_label.imgtk = imgtk
        self.image_label.configure(image=imgtk)
        self.result_text.configure(text=info_text)
        if self.is_live_running:
            self.root.after(30, self.show_frame)

    def start_live_analysis(self):
        """Canlı duygu analizini başlatır"""
        if not self.is_live_running:
            print("Canlı duygu analizi başlatıldı.")
            self.is_live_running = True
            self.cap = cv2.VideoCapture(0)
            self.live_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.image_button.config(state=tk.DISABLED)
            self.show_frame()
            
    def stop_live_analysis(self):
        """Canlı duygu analizini durdurur"""
        if self.is_live_running:
            print("Canlı duygu analizi durduruldu.")
            self.is_live_running = False
            if self.cap is not None:
                self.cap.release()
            self.cap = None
            self.live_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.image_button.config(state=tk.NORMAL)
            # Ekranı temizle
            self.image_label.config(image='')
            self.image_label.image = None
            self.result_text.config(text="")

    def analyze_image(self):
        print("Resimden duygu analizi başlatılıyor...")
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            try:
                file_path = os.path.abspath(file_path)
                print(f"Secilen resim yolu: {file_path}")
                image = cv2.imread(file_path)
                if image is None:
                    self.result_text.configure(text=turkce_to_ascii("Resim okunamadi! Lutfen gecerli bir resim dosyasi secin."))
                    print("Hata: Resim okunamadı veya geçersiz resim dosyası seçildi.")
                    return
                frame_width = self.image_frame.winfo_width()
                frame_height = self.image_frame.winfo_height()
                height, width = image.shape[:2]
                scale = min(frame_width/width, frame_height/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
                results = self.detector.detect_emotion(image)
                info_text = ""
                if results:
                    for result in results:
                        emotion = turkce_to_ascii(result['emotion'])
                        confidence = result['confidence']
                        # Mutlu duygusu için özel eşikler
                        if emotion == "Mutlu":
                            if confidence >= 0.25:  # Çok Mutlu için hassas eşik
                                emotion = "Çok Mutlu"
                            elif confidence >= 0.15:  # Mutlu için çok hassas eşik
                                emotion = "Mutlu"
                            else:
                                emotion = "Notr"
                        # Diğer duygular için kesin eşikler
                        elif emotion == "Uzgun":
                            if confidence > 0.85:  # Çok Üzgün için kesin eşik
                                emotion = "Çok Uzgun"
                            elif confidence > 0.70:  # Üzgün için kesin eşik
                                emotion = "Uzgun"
                            else:
                                emotion = "Notr"
                        elif emotion == "Kizgin":
                            if confidence > 0.85:  # Çok Kızgın için kesin eşik
                                emotion = "Çok Kizgin"
                            elif confidence > 0.70:  # Kızgın için kesin eşik
                                emotion = "Kizgin"
                            else:
                                emotion = "Notr"
                        elif emotion == "Korku":
                            if confidence > 0.85:  # Yoğun Korku için kesin eşik
                                emotion = "Çok Mutlu"
                            elif confidence > 0.70:  # Korku için kesin eşik
                                emotion = "Mutlu"
                            else:
                                emotion = "Notr"
                        elif emotion == "Igrenme":
                            if confidence > 0.85:  # Çok İğrenme için kesin eşik
                                emotion = "Çok Igrenme"
                            elif confidence > 0.70:  # İğrenme için kesin eşik
                                emotion = "Igrenme"
                            else:
                                emotion = "Notr"
                        elif emotion == "Saskin":
                            if confidence > 0.80:  # Çok Şaşkın için kesin eşik
                                emotion = "Çok Saskin"
                            elif confidence > 0.65:  # Şaşkın için kesin eşik
                                emotion = "Saskin"
                            else:
                                emotion = "Notr"
                        # Düşük güvenli tespitleri nötr olarak işaretle
                        elif confidence < 0.4:
                            emotion = "Notr"
                        bbox = result['bbox']
                        landmarks = result['landmarks']
                        analysis = result['landmark_analysis']
                        x, y, w, h = bbox
                        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        text = f"{emotion} ({confidence:.2f})"
                        cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        for px, py in landmarks:
                            cv2.circle(image, (px, py), 1, (255, 255, 255), -1)
                        analysis_text = f"Agiz: {analysis['mouth_ratio']:.2f}, Kas: {analysis['eyebrow_distance']:.2f}, Goz: {analysis['eye_ratio']:.2f}"
                        cv2.putText(image, analysis_text, (x, y+h+50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        self.db.save_emotion(emotion, confidence, confidence, source='image', image_path=file_path)
                        info_text = f"Duygu: {emotion}\nGuven: {confidence:.2f}"
                else:
                    info_text = turkce_to_ascii("Yuz tespit edilemedi!")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.image_label.imgtk = imgtk
                self.image_label.configure(image=imgtk)
                self.result_text.configure(text=info_text)
                records = self.db.get_image_records()
                self.update_statistics()
                print("Resim analizi tamamlandı.")
            except Exception as e:
                self.result_text.configure(text=turkce_to_ascii(f"Resim islenirken hata olustu: {str(e)}"))
                print(f"Hata: Resim işlenirken hata oluştu: {str(e)}")

    def run(self):
        self.root.mainloop()
        print("Uygulama kapatılıyor.")

    def create_tables(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS emotions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                emotion TEXT,
                confidence REAL,
                score REAL,
                source TEXT,
                image_path TEXT
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS emotion_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                emotion TEXT,
                confidence REAL,
                score REAL,
                timestamp TEXT
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS confidence_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                confidence REAL,
                timestamp TEXT
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS score_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                score REAL,
                timestamp TEXT
            )
        ''')
        self.conn.commit()

    def clear_database(self):
        """Veritabanını temizler"""
        if messagebox.askyesno("Onay", "Tüm veritabanı kayıtları silinecek. Emin misiniz?"):
            try:
                self.detector.db.clear_database()
                messagebox.showinfo("Başarılı", "Veritabanı başarıyla temizlendi.")
                print("Veritabanı başarıyla temizlendi.")
                # Arayüzü güncelle
                self.update_emotion_list()
                self.update_statistics()
                self.update_emotion_chart()
            except Exception as e:
                messagebox.showerror("Hata", f"Veritabanı temizlenirken hata oluştu: {str(e)}")
                print(f"Hata: Veritabanı temizlenirken hata oluştu: {str(e)}")

    def update_db_status(self):
        try:
            # Kayıt sayısını al
            self.cursor.execute("SELECT COUNT(*) FROM emotions")
            count = self.cursor.fetchone()[0]
            
            # Durum etiketini güncelle
            self.db_status_label.config(text=f"Veritabanında {count} kayıt bulunuyor")
        except Exception as e:
            self.db_status_label.config(text="Veritabanı durumu alınamadı")

    def show_statistics(self):
        """İstatistikleri gösterir"""
        try:
            records = self.detector.db.get_all_emotions()
            if records:
                from utils import calculate_statistics, plot_emotion_distribution
                emotions = [record[1] for record in records]  # Duygu sütunu
                calculate_statistics(records)
                plot_emotion_distribution(emotions)
            else:
                messagebox.showinfo("Bilgi", "Gösterilecek kayıt bulunamadı.")
        except Exception as e:
            messagebox.showerror("Hata", f"İstatistikler gösterilirken hata oluştu: {str(e)}")

    def update_emotion_list(self):
        """Duygu listesini günceller"""
        for item in self.tree.get_children():
            self.tree.delete(item)
        records = self.detector.db.get_all_emotions()
        for record in records:
            self.tree.insert('', 'end', values=record)

    def update_emotion_chart(self):
        """Duygu grafiklerini günceller"""
        records = self.detector.db.get_all_emotions()
        if records:
            from utils import plot_emotion_distribution
            emotions = [record[1] for record in records]
            plot_emotion_distribution(emotions)
        else:
            print("Güncellenecek grafik verisi bulunamadı.")

    def on_closing(self):
        # Konsol çıktısını geri yükle
        sys.stdout = self.console_redirector.original_stdout

        # Çıktıları bir dosyaya kaydet
        output_filename = "analiz_sonuclari.txt"
        
        # Güncel tarih ve saati al ve formatla (baslangic ve bitis icin)
        now = datetime.now()
        timestamp_header_start = now.strftime("%d %m %Y %H:%M:%S")
        
        # Dosya içeriğine başlık olarak ekle
        file_content = f"\n\n--- Analiz Baslangic Zamani: {timestamp_header_start} ---\n\n" \
                       + self.console_redirector.get_output()
                       
        # Analiz bitis zamanini ekle
        timestamp_header_end = datetime.now().strftime("%d %m %Y %H:%M:%S")
        file_content += f"\n\n--- Analiz Bitis Zamani: {timestamp_header_end} ---\n"

        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(file_content)
        print(f"Konsol ciktilari '{output_filename}' dosyasina kaydedildi.")
        
        # Pencereyi kapat
        self.root.destroy()

if __name__ == "__main__":
    app = EmotionAnalysisApp()
    app.run() 