import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import os

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
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 38 * 38, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# --- УНІВЕРСАЛЬНЕ НАЛАШТУВАННЯ ШЛЯХІВ ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

DATA_DIR = os.path.join(project_root, 'data')
MODEL_PATH = os.path.join(project_root, 'sound_model.pth')
DATASET_PATH = os.path.join(DATA_DIR, 'dataset')

os.chdir(project_root)
print(f"Проект знаходиться в: {project_root}")
# ---------------------------------------

try:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Використовую пристрій: {DEVICE}")
    
    model = SoundClassifier().to(DEVICE)
    
    if not os.path.exists(MODEL_PATH):
        print(f"ПОМИЛКА: Файл {MODEL_PATH} не знайдено!")
    else:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Модель завантажена успішно")
    
    model.train()
    
    EPOCHS = 7
    LR = 0.0001
    
    transform = transforms.Compose([
        transforms.Resize((155, 154)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    if not os.path.exists(DATASET_PATH):
        print(f"ПОМИЛКА: Папка dataset не знайдена за шляхом: {DATASET_PATH}")
    else:
        train_data = datasets.ImageFolder(DATASET_PATH, transform=transform)
        print(f"Знайдено {len(train_data)} зображень у класах: {train_data.classes}")
        
        if len(train_data) == 0:
            print("ПОМИЛКА: Dataset порожній!")
        else:
            train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=LR)
            
            print("Починаю донавчання...")
            for epoch in range(EPOCHS):
                epoch_loss = 0
                for batch_idx, (images, labels) in enumerate(train_loader):
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    
                    if batch_idx % 10 == 0:
                        print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
                
                print(f"Епоха {epoch+1}/{EPOCHS} завершена. Середня втрата: {epoch_loss/len(train_loader):.4f}")
            
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Модель оновлена та збережена: {MODEL_PATH}")

except Exception as e:
    print(f"ПОМИЛКА: {type(e).__name__}")
    print(f"Деталі: {str(e)}")
    import traceback
    traceback.print_exc()