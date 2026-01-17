import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# 1. Налаштування папок (Універсальний шлях для будь-якого ПК)
# Отримуємо шлях до 'scripts', а потім піднімаємось на рівень вгору до кореня проекту
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'data', 'dataset')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Підготовка картинок
transform = transforms.Compose([
    transforms.Resize((155, 154)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Завантаження датасету
if not os.path.exists(DATASET_PATH):
    print(f"Помилка: Папка не знайдена за шляхом: {DATASET_PATH}")
    exit() # Зупиняємо скрипт, якщо даних немає
else:
    dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(f"Успішно завантажено! Класи: {dataset.classes}")

# 3. Архітектура нейромережі (Проста CNN)
class SoundClassifier(nn.Module):
    def __init__(self):
        super(SoundClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )
        # Розраховано для входу 155x154
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 38 * 38, 128), nn.ReLU(),
            nn.Linear(128, len(dataset.classes))
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# 4. Процес навчання
model = SoundClassifier().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n--- Починаємо навчання (20 епох) ---")
for epoch in range(20):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Епоха {epoch+1}/20 | Помилка: {running_loss/len(train_loader):.4f}")

# 5. Збереження результату
torch.save(model.state_dict(), "sound_model.pth")
print("\nУспіх! Модель збережена як 'sound_model.pth'")