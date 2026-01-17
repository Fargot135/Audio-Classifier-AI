import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def create_images(file_path, base_dir):
    folder_name = os.path.splitext(os.path.basename(file_path))[0]
    save_dir = os.path.join(base_dir, folder_name)

    if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0:
        return

    os.makedirs(save_dir, exist_ok=True)
    print(f"+++ Обробка: {folder_name}...")

    try:
        # Завантажуємо аудіо
        y, sr = librosa.load(file_path)
        duration = int(librosa.get_duration(y=y, sr=sr))
        
        # Крок 3 секунди (0, 3, 6...)
        for i in range(0, duration, 3):
            start, end = i * sr, (i + 3) * sr
            chunk = y[start:end]
            
            # Перевірка: фрагмент має бути рівно 3 секунди
            if len(chunk) < 3 * sr: 
                continue
            
            # Створення мел-спектрограми
            S = librosa.feature.melspectrogram(y=chunk, sr=sr)
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            # Малювання та збереження
            plt.figure(figsize=(2, 2))
            librosa.display.specshow(S_dB, sr=sr)
            plt.axis('off')
            plt.savefig(f"{save_dir}/img_{i}.png", bbox_inches='tight', pad_inches=0)
            plt.close()
            
    except Exception as e:
        print(f"Помилка у файлі {file_path}: {e}")

def process_category(source_folder, dataset_path):
    abs_path = os.path.abspath(source_folder)
    
    if not os.path.exists(source_folder):
        print(f"--- Папка не знайдена: {abs_path}")
        return

    files = [f for f in os.listdir(source_folder) if f.lower().strip().endswith('.wav')]
    print(f"Папка {source_folder}: знайдено {len(files)} файлів")

    for f in files:
        full_path = os.path.join(source_folder, f)
        create_images(full_path, dataset_path)

if __name__ == "__main__":
    # 1. Отримуємо шлях до папки, де лежить скрипт (scripts)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Переходимо на рівень вгору до головної папки проекту (Second Git)
    project_root = os.path.dirname(current_dir)
    
    # 3. Шлях до папки з даними
    base_path = os.path.join(project_root, 'data')

    print(f"Коренева папка проекту: {project_root}")
    print(f"Шукаю дані в: {base_path}")

    # Обробка категорій
    process_category(os.path.join(base_path, 'music_wav'), os.path.join(base_path, 'dataset', 'music'))
    process_category(os.path.join(base_path, 'noise_wav'), os.path.join(base_path, 'dataset', 'noise'))
    
    print("\n--- Готово! ---")