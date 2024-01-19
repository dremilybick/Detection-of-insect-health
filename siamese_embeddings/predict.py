import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SiameseNetwork, TripletLoss
from dataloader import SiameseDataset, SiamesePredictionDataset, TripletSampler
from torch.utils.tensorboard import SummaryWriter
from utils import load_data
import numpy as np
import pandas as pd

run_id = 5
model_ckpt = f'runs/siamese_run_{run_id}/siamese_model.pth'

input_size = 400
embedding_size = 64
batch_size = 16

#%% Save predictions

predict_signals, predict_thresholds, _ = load_data('train.txt')
train_meta = pd.read_csv('train.txt')
predict_dataset = SiamesePredictionDataset(predict_signals, predict_thresholds)
predict_loader = DataLoader(predict_dataset, batch_size=32)

checkpoint = torch.load(model_ckpt)

# Load the model state_dict
model = SiameseNetwork(input_size, embedding_size)
model.load_state_dict(checkpoint)

model.eval()
embeddings = pd.DataFrame()
with torch.no_grad():
    for batch in predict_loader:
        batch_size = len(batch)
        batch = batch.reshape(batch_size, 1, 400)
        embedding = model(batch)
        embedding = embedding.detach().numpy()
        embedding = pd.DataFrame(embedding)
        embeddings = pd.concat([embeddings, embedding])

embeddings = embeddings.reset_index(drop=True)

embeddings = pd.concat([train_meta, embeddings], axis=1)

embeddings.to_csv(f'runs/siamese_run_{run_id}/embeddings.csv', index=False)