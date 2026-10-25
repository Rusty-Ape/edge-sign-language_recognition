import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

#################################################
# LOAD DATA
#################################################

data = np.load("asl_landmarks.npz", allow_pickle=True)

X = data["X"]
y = data["y"]
classes = data["classes"]

num_classes = len(classes)

#################################################
# DATASET
#################################################

class SignDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = SignDataset(X, y)

#################################################
# SPLIT
#################################################

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

#################################################
# MODEL
#################################################

class SignMLP(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(63, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = SignMLP(num_classes)

#################################################
# TRAINING SETUP
#################################################

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#################################################
# TRAIN LOOP
#################################################

for epoch in range(25):

    model.train()

    correct, total, loss_sum = 0, 0, 0

    for X_batch, y_batch in train_loader:

        optimizer.zero_grad()

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

        preds = outputs.argmax(1)

        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    train_acc = correct / total

    #################################################

    model.eval()

    correct, total = 0, 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:

            outputs = model(X_batch)
            preds = outputs.argmax(1)

            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    test_acc = correct / total

    print(
        f"Epoch {epoch+1} | "
        f"Loss {loss_sum:.4f} | "
        f"Train Acc {train_acc:.3f} | "
        f"Test Acc {test_acc:.3f}"
    )

#################################################
# SAVE MODEL
#################################################

torch.save({
    "model_state_dict": model.state_dict(),
    "classes": classes
}, "sign_mlp.pth")

print("Model saved → sign_mlp.pth")