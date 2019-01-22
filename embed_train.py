import torch
import pandas as pd
from torch import nn, optim
from tabular import TabularDataset, FeedForwardNN, cat_lookup
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

# Using only a subset of the variables.
train_data = pd.read_csv("misc/new_train.csv",
                         usecols=["Type", "Age", "Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3",
                                  "MaturitySize",
                                  "FurLength", "Vaccinated", "Dewormed", "Sterilized", "Health", "Quantity", "Fee",
                                  "State",
                                  "AdoptionSpeed"]).dropna()
val_data = pd.read_csv("misc/new_val.csv",
                       usecols=["Type", "Age", "Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3",
                                "MaturitySize",
                                "FurLength", "Vaccinated", "Dewormed", "Sterilized", "Health", "Quantity", "Fee",
                                "State",
                                "AdoptionSpeed"]).dropna()
categorical_features = ["Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3", "MaturitySize",
                        "FurLength", "Vaccinated", "Dewormed", "Sterilized", "Health", "State"]
output_feature = "AdoptionSpeed"

label_encoders = {}
for cat_col in categorical_features:
    if cat_col not in ["Breed1", "Breed2", "State", "Gender", "Vaccinated", "Dewormed", "Sterilized"]:
        continue
    train_data[cat_col] = cat_lookup(train_data[cat_col], cat_col)
    val_data[cat_col] = cat_lookup(val_data[cat_col], cat_col)
train_dataset = TabularDataset(data=train_data, cat_cols=categorical_features,
                               output_col=output_feature)
val_dataset = TabularDataset(data=train_data, cat_cols=categorical_features)
batchsize = 64
train_dataloader = DataLoader(train_dataset, batchsize, shuffle=True, num_workers=8)
val_dataloader = DataLoader(val_dataset, 1, shuffle=False, num_workers=8)

cat_dims = [308, 308, 3, 8, 8, 8, 5, 4, 3, 3, 3, 4, 15]
emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]

device = 'cuda'
model = FeedForwardNN(emb_dims, no_of_cont=4, lin_layer_sizes=[50, 100],
                      output_size=5, emb_dropout=0.04,
                      lin_layer_dropouts=[0.001, 0.01]).to(device)
no_of_epochs = 100
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
schedule = ReduceLROnPlateau(optimizer=optimizer,
                             mode='max')

while True:
    total = 0
    correct = 0
    val_total = 0
    val_correct = 0
    best_acc = 0
    model.train()
    for idx, (y, cont_x, cat_x) in enumerate(train_dataloader):
        cat_x = cat_x.to(device)
        cont_x = cont_x.to(device)
        y = y.to(device)
        preds = model(cont_x, cat_x)
        loss = criterion(preds, y.long())
        _, predicted = preds.max(1)
        total += y.size(0)
        correct += predicted.eq(y.long()).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(idx, end='\r')
    model.eval()
    for idx, (y, cont_x, cat_x) in enumerate(val_dataloader):
        cat_x = cat_x.to(device)
        cont_x = cont_x.to(device)
        y = y.to(device)
        with torch.no_grad():
            preds = model(cont_x, cat_x)
            _, predicted = preds.max(1)
            val_total += y.size(0)
            val_correct += predicted.eq(y.long()).sum().item()

    schedule.step(correct)
    print(f'acc: {correct/total:.4} - val_acc:{val_correct/val_total:.4}')
    state = {
        'optimizer': optimizer.state_dict(),
        'net': model.state_dict(),
        'acc': val_correct / val_total,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if val_correct > best_acc:
        torch.save(state, f'./checkpoint/embeded1_best.t7')
        best_acc = val_correct
    torch.save(state, f'./checkpoint/embeded1_temp.t7')
