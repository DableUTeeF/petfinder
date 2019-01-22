import torch
import pandas as pd
from torch import nn, optim
from tabular import TabularDataset, FeedForwardNN, cat_lookup
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


# Using only a subset of the variables.
csv = open('misc/test.csv', 'r').readlines()
data = pd.read_csv("misc/test.csv",
                   usecols=["Type", "Age", "Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3", "MaturitySize",
                            "FurLength", "Vaccinated", "Dewormed", "Sterilized", "Health", "Quantity", "Fee", "State"]).dropna()
categorical_features = ["Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3", "MaturitySize",
                        "FurLength", "Vaccinated", "Dewormed", "Sterilized", "Health", "State"]
output_feature = "AdoptionSpeed"

label_encoders = {}
for cat_col in categorical_features:
    if cat_col not in ["Breed1", "Breed2", "State", "Gender", "Vaccinated", "Dewormed", "Sterilized"]:
        continue
    data[cat_col] = cat_lookup(data[cat_col], cat_col)
dataset = TabularDataset(data=data, cat_cols=categorical_features)
batchsize = 64
dataloader = DataLoader(dataset, 1, shuffle=True, num_workers=8)

cat_dims = [308, 308, 3, 8, 8, 8, 5, 4, 3, 3, 3, 4, 15]
emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]

device = 'cuda'
model = FeedForwardNN(emb_dims, no_of_cont=4, lin_layer_sizes=[50, 100],
                      output_size=5, emb_dropout=0.04,
                      lin_layer_dropouts=[0.001, 0.01]).to(device)
model.eval()
no_of_epochs = 100
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
schedule = ReduceLROnPlateau(optimizer=optimizer,
                             mode='max')
checkpoint = torch.load(f'./checkpoint/embeded1_best.t7')
model.load_state_dict(checkpoint['net'])
total = 0
correct = 0
val_total = 0
val_correct = 0
best_acc = 0
with open('prd/submission.csv', 'w') as wr:
    wr.write('PetID,AdoptionSpeed\n')
    for idx, (_, cont_x, cat_x) in enumerate(dataloader):
        try:
            pid = csv[idx+1].split(',')[-2]
            cat_x = cat_x.to(device)
            cont_x = cont_x.to(device)
            with torch.no_grad():
                preds = model(cont_x, cat_x)
                _, predicted = preds.max(1)
                wr.write(f'{pid},'+str(predicted.cpu().detach().numpy()[0])+'\n')
        except TypeError:
            print(idx)
