# -*- coding: utf-8 -*-
"""
Распознавание рукописных цифр с возможностью дообучения.
Рисуй мышкой цифру, нажимай "Распознать", если ошибка – вводи правильную цифру и жми "Обучить".
Модель: простая CNN, предобученная на MNIST.
"""

import tkinter as tk
from tkinter import messagebox, simpledialog
import numpy as np
from PIL import Image, ImageDraw, ImageGrab
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# ------------------------- Модель -------------------------
def create_model():
    """Создаёт свёрточную модель для MNIST (28x28, 1 канал)."""
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

MODEL_FILE = 'mnist_cnn.keras'

# Если файл модели существует – загружаем, иначе обучаем с нуля на MNIST
if os.path.exists(MODEL_FILE):
    print("Загрузка сохранённой модели...")
    model = keras.models.load_model(MODEL_FILE)
else:
    print("Модель не найдена. Обучаем на MNIST (это займёт пару минут)...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # Нормализация и добавление канала
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    model = create_model()
    model.fit(x_train, y_train, batch_size=128, epochs=3, validation_split=0.1)
    # Оцениваем на тесте
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Точность на тесте: {test_acc:.4f}")
    model.save(MODEL_FILE)
    print(f"Модель сохранена в {MODEL_FILE}")

# ------------------------- Функции обработки изображения -------------------------
def prepare_image(img):
    """
    Принимает PIL Image (RGB) с рисунком на белом фоне,
    преобразует в чёрно-белое 28x28, инвертирует (чёрный фон, белая цифра)
    и нормализует.
    """
    # Переводим в градации серого
    img = img.convert('L')
    # Инвертируем: белый фон (255) станет 0, чёрная линия (0) станет 255
    img = Image.eval(img, lambda x: 255 - x)
    # Масштабируем до 28x28
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    # Преобразуем в numpy массив [0-255] и нормализуем в [0-1]
    arr = np.array(img, dtype=np.float32) / 255.0
    # Добавляем размерность канала (28,28,1)
    arr = np.expand_dims(arr, axis=-1)
    # Добавляем размерность батча (1,28,28,1)
    arr = np.expand_dims(arr, axis=0)
    return arr

# ------------------------- Интерфейс -------------------------
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Рисуй цифру и учи нейросеть")
        self.root.geometry("400x500")
        self.root.resizable(False, False)

        # Холст для рисования
        self.canvas = tk.Canvas(root, width=280, height=280, bg='white', bd=3, relief=tk.SUNKEN)
        self.canvas.pack(pady=10)

        # Привязка событий мыши
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)
        self.last_x, self.last_y = None, None

        # Кнопки
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)

        self.recognize_btn = tk.Button(btn_frame, text="Распознать", command=self.recognize, width=12)
        self.recognize_btn.grid(row=0, column=0, padx=5)

        self.clear_btn = tk.Button(btn_frame, text="Очистить", command=self.clear_canvas, width=12)
        self.clear_btn.grid(row=0, column=1, padx=5)

        self.train_btn = tk.Button(btn_frame, text="Обучить на цифре...", command=self.train_on_drawing, width=20)
        self.train_btn.grid(row=1, column=0, columnspan=2, pady=5)

        # Метка для вывода результата
        self.result_label = tk.Label(root, text="Нарисуй цифру и нажми Распознать", font=('Arial', 14))
        self.result_label.pack(pady=10)

        # Для сохранения последнего рисунка (чтобы не перерисовывать)
        self.last_image = None

    def paint(self, event):
        """Рисование мышкой."""
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, x, y, width=15, fill='black', capstyle=tk.ROUND, smooth=True)
        self.last_x, self.last_y = x, y

    def reset_coords(self, event):
        """Сброс координат (можно не использовать, но добавим для отвязки)."""
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        """Очистить холст."""
        self.canvas.delete("all")
        self.last_x, self.last_y = None, None
        self.result_label.config(text="Холст очищен")

    def get_canvas_image(self):
        """Захватывает содержимое холста как PIL Image."""
        # Координаты холста на экране
        x = self.canvas.winfo_rootx()
        y = self.canvas.winfo_rooty()
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        # Захват области холста
        img = ImageGrab.grab(bbox=(x, y, x+w, y+h))
        return img

    def recognize(self):
        """Распознать цифру на холсте."""
        img = self.get_canvas_image()
        self.last_image = img.copy()  # сохраняем для обучения
        processed = prepare_image(img)
        pred = model.predict(processed, verbose=0)
        digit = np.argmax(pred[0])
        confidence = pred[0][digit]
        self.result_label.config(text=f"Это цифра: {digit} (уверенность: {confidence:.2f})")

    def train_on_drawing(self):
        """Дообучить модель на текущем рисунке с указанной пользователем цифрой."""
        if self.last_image is None:
            messagebox.showwarning("Нет рисунка", "Сначала нажми 'Распознать', чтобы захватить рисунок.")
            return

        # Запрашиваем правильную цифру
        correct = simpledialog.askinteger("Обучение", "Введите правильную цифру (0-9):",
                                          minvalue=0, maxvalue=9)
        if correct is None:
            return

        # Подготавливаем изображение
        x = prepare_image(self.last_image)  # уже с батчем (1,28,28,1)
        y = np.array([correct], dtype=np.int32)

        # Один шаг обучения
        loss, acc = model.train_on_batch(x, y)
        # Сохраняем обновлённую модель
        model.save(MODEL_FILE)
        self.result_label.config(text=f"Дообучено на цифре {correct}. Потери: {loss:.4f}")

        # Небольшое сообщение
        messagebox.showinfo("Готово", f"Модель обновилась на примере цифры {correct}.")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    # При отпускании кнопки мыши сбрасываем координаты, чтобы линии не соединялись через пустое пространство
    root.bind("<ButtonRelease-1>", lambda e: app.reset_coords(e))
    root.mainloop()