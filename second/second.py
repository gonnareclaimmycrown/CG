import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Обработка изображений - Пороговая обработка и фильтры")
        self.root.geometry("1200x700")
        
        self.original_image = None
        self.processed_image = None
        self.thresholded_image = None
        self.filtered_image = None
        self.test_images = {}
        
        self.original_photo = None
        self.processed_photo = None
        
        self.setup_ui()
        self.generate_test_images()
        self.root.after(100, self.load_test_image)
    
    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        control_frame = ttk.LabelFrame(main_frame, text="Управление", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        ttk.Label(control_frame, text="Тестовые изображения:").pack(anchor=tk.W)
        self.image_var = tk.StringVar()
        test_images = ["Зашумленное", "Размытое", "Малоконтрастное", 
                    "Текст на неровном фоне", "Объекты разной яркости"]
        self.image_combo = ttk.Combobox(control_frame, textvariable=self.image_var, 
                                    values=test_images, state="readonly")
        self.image_combo.set("Зашумленное")
        self.image_combo.pack(fill=tk.X, pady=(0, 10))
        self.image_combo.bind('<<ComboboxSelected>>', self.load_test_image)
        
        threshold_frame = ttk.LabelFrame(control_frame, text="Глобальная пороговая обработка (2 метода)", padding=5)
        threshold_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.threshold_var = tk.StringVar(value="Метод Оцу")
        ttk.Radiobutton(threshold_frame, text="Метод Оцу", variable=self.threshold_var, 
                    value="Метод Оцу").pack(anchor=tk.W)
        ttk.Radiobutton(threshold_frame, text="Адаптивный метод", variable=self.threshold_var, 
                    value="Адаптивный метод").pack(anchor=tk.W)
        
        filter_frame = ttk.LabelFrame(control_frame, text="Низкочастотные фильтры", padding=5)
        filter_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.filter_var = tk.StringVar(value="Гауссовский")
        filters = ["Гауссовский", "Медианный", "Усредняющий"]
        for filter_name in filters:
            ttk.Radiobutton(filter_frame, text=filter_name, variable=self.filter_var, 
                    value=filter_name).pack(anchor=tk.W)
        
        buttons_frame = ttk.LabelFrame(control_frame, text="Операции обработки", padding=5)
        buttons_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(buttons_frame, text="Только пороговая обработка", 
                command=self.apply_threshold_only).pack(fill=tk.X, pady=2)
        ttk.Button(buttons_frame, text="Только фильтр", 
                command=self.apply_filter_only).pack(fill=tk.X, pady=2)
        ttk.Button(buttons_frame, text="Порог + Фильтр (последовательно)", 
                command=self.process_image).pack(fill=tk.X, pady=2)
        
        ttk.Button(control_frame, text="Загрузить своё изображение", 
                command=self.load_custom_image).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Сравнить методы пороговой обработки", 
                command=self.compare_methods).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Показать гистограмму", 
                command=self.show_histogram).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Сбросить обработку", 
                command=self.reset_processing).pack(fill=tk.X, pady=5)
        
        self.original_label = ttk.Label(image_frame, text="Исходное изображение")
        self.original_label.pack()
        self.original_canvas = tk.Canvas(image_frame, width=400, height=300, bg="white")
        self.original_canvas.pack(pady=(0, 20))
        
        self.processed_label = ttk.Label(image_frame, text="Обработанное изображение")
        self.processed_label.pack()
        self.processed_canvas = tk.Canvas(image_frame, width=400, height=300, bg="white")
        self.processed_canvas.pack()
    
    def generate_test_images(self):
        
        noise_img = np.random.randint(0, 255, (300, 400), dtype=np.uint8)
        cv2.circle(noise_img, (200, 150), 80, 150, -1)
        cv2.rectangle(noise_img, (50, 50), (150, 100), 200, -1)
        self.test_images["Зашумленное"] = noise_img
        
        blur_img = np.zeros((300, 400), dtype=np.uint8)
        cv2.rectangle(blur_img, (50, 50), (350, 250), 200, -1)
        cv2.circle(blur_img, (200, 150), 60, 150, -1)
        self.test_images["Размытое"] = cv2.GaussianBlur(blur_img, (25, 25), 10)
        
        low_contrast = np.full((300, 400), 100, dtype=np.uint8)
        cv2.rectangle(low_contrast, (100, 100), (300, 200), 120, -1)
        cv2.circle(low_contrast, (200, 150), 30, 80, -1)
        self.test_images["Малоконтрастное"] = low_contrast
        
        text_img = np.zeros((300, 400), dtype=np.uint8)
        for i in range(400):
            text_img[:, i] = 50 + int(100 * i / 400)
        noise = np.random.randint(0, 30, (300, 400), dtype=np.uint8)
        text_img = cv2.add(text_img, noise)
        cv2.putText(text_img, "Hello World!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, 200, 3)
        cv2.putText(text_img, "Adaptive vs Otsu", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 180, 2)
        self.test_images["Текст на неровном фоне"] = text_img
        

        objects_img = np.full((300, 400), 100, dtype=np.uint8)
        cv2.circle(objects_img, (100, 100), 30, 50, -1)
        cv2.circle(objects_img, (200, 100), 30, 100, -1)
        cv2.circle(objects_img, (300, 100), 30, 150, -1)
        cv2.rectangle(objects_img, (150, 200), (250, 250), 200, -1)
        self.test_images["Объекты разной яркости"] = objects_img
    
    def load_test_image(self, event=None):
        image_name = self.image_var.get()
        if image_name in self.test_images:
            self.original_image = self.test_images[image_name]
            self.display_image(self.original_image, self.original_canvas)
            self.processed_canvas.delete("all")
            self.processed_label.config(text="Обработанное изображение")
    
    def load_custom_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self.original_image = img
                self.display_image(self.original_image, self.original_canvas)
                self.processed_canvas.delete("all")
                self.processed_label.config(text="Обработанное изображение")
            else:
                messagebox.showerror("Ошибка", "Не удалось загрузить изображение")
    
    def display_image(self, image, canvas):
        if image is None:
            return
            
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        pil_image = Image.fromarray(image_rgb)
        
        canvas.update_idletasks()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1:
            canvas_width = 400
        if canvas_height <= 1:
            canvas_height = 300
        
        img_ratio = pil_image.width / pil_image.height
        canvas_ratio = canvas_width / canvas_height
        
        if img_ratio > canvas_ratio:
            new_width = canvas_width
            new_height = int(canvas_width / img_ratio)
        else:
            new_height = canvas_height
            new_width = int(canvas_height * img_ratio)
        
        new_width = max(1, new_width)
        new_height = max(1, new_height)
        
        try:
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            if canvas == self.original_canvas:
                self.original_photo = ImageTk.PhotoImage(pil_image)
                photo = self.original_photo
            else:
                self.processed_photo = ImageTk.PhotoImage(pil_image)
                photo = self.processed_photo
            
            canvas.delete("all")
            canvas.create_image(canvas_width//2, canvas_height//2, anchor=tk.CENTER, image=photo)
        except Exception as e:
            print(f"Ошибка отображения изображения: {e}")
    
    def apply_threshold_only(self):
        if self.original_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение")
            return
        
        try:
            method = self.threshold_var.get()
            
            if method == "Метод Оцу":
                _, self.thresholded_image = cv2.threshold(self.original_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                method_name = "Метод Оцу"
            
            elif method == "Адаптивный метод":
                self.thresholded_image = cv2.adaptiveThreshold(
                    self.original_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
                method_name = "Адаптивный метод"
            
            self.processed_image = self.thresholded_image
            self.display_image(self.processed_image, self.processed_canvas)
            self.processed_label.config(text=f"Только пороговая обработка ({method_name})")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка пороговой обработки: {str(e)}")
    
    def apply_filter_only(self):
        if self.original_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение")
            return
        
        try:
            method = self.filter_var.get()
            
            if method == "Гауссовский":
                self.filtered_image = cv2.GaussianBlur(self.original_image, (5, 5), 0)
            elif method == "Медианный":
                self.filtered_image = cv2.medianBlur(self.original_image, 5)
            elif method == "Усредняющий":
                kernel = np.ones((5, 5), np.float32) / 25
                self.filtered_image = cv2.filter2D(self.original_image, -1, kernel)
            
            self.processed_image = self.filtered_image
            self.display_image(self.processed_image, self.processed_canvas)
            self.processed_label.config(text=f"Только фильтр ({method})")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка применения фильтра: {str(e)}")
    
    def process_image(self):
        if self.original_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение")
            return
        
        try:
            method = self.threshold_var.get()
            
            if method == "Метод Оцу":
                _, thresholded = cv2.threshold(self.original_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                threshold_name = "Метод Оцу"
            
            elif method == "Адаптивный метод":
                thresholded = cv2.adaptiveThreshold(
                    self.original_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
                threshold_name = "Адаптивный метод"
            
            filter_method = self.filter_var.get()
            
            if filter_method == "Гауссовский":
                filtered = cv2.GaussianBlur(thresholded, (5, 5), 0)
            elif filter_method == "Медианный":
                filtered = cv2.medianBlur(thresholded, 5)
            elif filter_method == "Усредняющий":
                kernel = np.ones((5, 5), np.float32) / 25
                filtered = cv2.filter2D(thresholded, -1, kernel)
            
            self.processed_image = filtered
            self.display_image(self.processed_image, self.processed_canvas)
            self.processed_label.config(text=f"Порог ({threshold_name}) + Фильтр ({filter_method})")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка обработки: {str(e)}")
    
    def reset_processing(self):
        if self.original_image is not None:
            self.processed_canvas.delete("all")
            self.processed_label.config(text="Обработанное изображение (сброс)")
    
    def compare_methods(self):
        if self.original_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение")
            return
        
        compare_window = tk.Toplevel(self.root)
        compare_window.title("Сравнение методов пороговой обработки")
        compare_window.geometry("1000x600")
        
        otsu_result, _ = self.apply_threshold_method_otsu(self.original_image)
        adaptive_result, _ = self.apply_threshold_method_adaptive(self.original_image)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        ax1.imshow(self.original_image, cmap='gray')
        ax1.set_title('Исходное изображение')
        ax1.axis('off')
        
        ax2.imshow(otsu_result, cmap='gray')
        ax2.set_title('Метод Оцу')
        ax2.axis('off')
        
        ax3.imshow(adaptive_result, cmap='gray')
        ax3.set_title('Адаптивный метод')
        ax3.axis('off')
        
        ax4.hist(self.original_image.ravel(), bins=256, range=[0, 256], alpha=0.7, color='blue')
        ax4.set_title('Гистограмма исходного изображения')
        ax4.set_xlabel('Яркость')
        ax4.set_ylabel('Частота')
        
        otsu_threshold = cv2.threshold(self.original_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        ax4.axvline(x=otsu_threshold, color='red', linestyle='--', label=f'Порог Оцу: {otsu_threshold:.1f}')
        ax4.legend()
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=compare_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        explanation = """
        Сравнение методов:
        • Метод Оцу: Лучше работает на изображениях с равномерным освещением и четким разделением гистограммы
        • Адаптивный метод: Лучше на изображениях с неравномерным освещением (текст, тени, градиенты)
        """
        
        label = ttk.Label(compare_window, text=explanation, justify=tk.LEFT)
        label.pack(pady=10)
    
    def apply_threshold_method_otsu(self, image):
        _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresholded, "Метод Оцу"
    
    def apply_threshold_method_adaptive(self, image):
        adaptive_thresh = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return adaptive_thresh, "Адаптивный метод"
    
    def show_histogram(self):
        if self.original_image is None:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        ax1.hist(self.original_image.ravel(), bins=256, range=[0, 256], alpha=0.7)
        ax1.set_title('Гистограмма исходного изображения')
        ax1.set_xlabel('Значение пикселя')
        ax1.set_ylabel('Частота')
        
        otsu_threshold = cv2.threshold(self.original_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        ax1.axvline(x=otsu_threshold, color='red', linestyle='--', 
            label=f'Порог Оцу: {otsu_threshold:.1f}')
        ax1.legend()
        
        if self.processed_image is not None:
            ax2.hist(self.processed_image.ravel(), bins=256, range=[0, 256], alpha=0.7, color='orange')
            ax2.set_title('Гистограмма обработанного изображения')
            ax2.set_xlabel('Значение пикселя')
            ax2.set_ylabel('Частота')
        
        plt.tight_layout()
        
        hist_window = tk.Toplevel(self.root)
        hist_window.title("Гистограммы изображений")
        
        canvas = FigureCanvasTkAgg(fig, master=hist_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def main():
    root = tk.Tk()
    app = ImageProcessor(root)
    root.mainloop()

if __name__ == "__main__":
    main()