import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SiameseNetwork, TripletLoss
from dataloader import SiameseDataset, SiamesePredictionDataset, TripletSampler
from torch.utils.tensorboard import SummaryWriter
from utils import load_data
import os

run_id = 0
while os.path.exists(f'runs/siamese_run_{run_id}'):
    run_id += 1
save_folder = f'runs/siamese_run_{run_id}'

writer = SummaryWriter(save_folder)

# Initialize your Siamese network and other training components
input_size = 400
embedding_size = 64
batch_size = 16
initial_lr = 0.0001
num_epochs = 50
scheduler_patience = 3
loss_margin = 1

checkpoint = f'runs/siamese_run_5/siamese_model.pth'
#checkpoint = None

siamese_model = SiameseNetwork(input_size, embedding_size)

if checkpoint is not None:
    checkpoint = torch.load(checkpoint)
    siamese_model.load_state_dict(checkpoint)

triplet_loss = TripletLoss(margin=loss_margin)
optimiser = optim.Adam(siamese_model.parameters(), lr=initial_lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=scheduler_patience)

train_signals, train_thresholds, train_origin = load_data('train.txt')
test_signals, test_thresholds, test_origin = load_data('test.txt')

train_dataset = SiameseDataset(train_signals, train_thresholds, train_origin)
test_dataset = SiameseDataset(test_signals, test_thresholds, test_origin)

triplet_train = TripletSampler(train_dataset, siamese_model, batch_size=batch_size)
train_loader = DataLoader(train_dataset, batch_sampler=triplet_train)
triplet_test = TripletSampler(test_dataset, siamese_model, batch_size=batch_size)
test_loader = DataLoader(train_dataset, batch_sampler=triplet_train)

for epoch in range(num_epochs):
    siamese_model.train()
    avg_loss = 0.0
    for batch in train_loader:
        anchor, positive, negative = batch

        batch_size = len(anchor)
        anchor = anchor.reshape(batch_size,1, input_size)
        positive = positive.reshape(batch_size,1, input_size)
        negative = negative.reshape(batch_size, 1,input_size)

        # Forward pass
        anchor_embedding = siamese_model(anchor)
        positive_embedding = siamese_model(positive)
        negative_embedding = siamese_model(negative)

        # Calculate triplet loss
        loss = triplet_loss(anchor_embedding, positive_embedding, negative_embedding)
        avg_loss += loss

        # Backward pass and optimisation
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    for name, param in siamese_model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(f'gradients/{name}', param.grad, epoch)

    writer.add_scalar('training loss',
                      loss.item()/batch_size,
                      epoch + 1)
    writer.add_scalar('average training loss',
                      avg_loss.item()/len(train_loader),
                      epoch + 1)

    # Validation
    siamese_model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for batch in test_loader:
            anchor, positive, negative = batch

            batch_size = len(anchor)
            anchor = anchor.reshape(batch_size, 1, input_size)
            positive = positive.reshape(batch_size, 1, input_size)
            negative = negative.reshape(batch_size, 1, input_size)

            # Forward pass
            anchor_embedding = siamese_model(anchor)
            positive_embedding = siamese_model(positive)
            negative_embedding = siamese_model(negative)
            test_loss += triplet_loss(anchor_embedding, positive_embedding, negative_embedding)

    num_samples = len(test_loader)
    test_loss /= num_samples
    writer.add_scalar('holdout loss',
                      test_loss.item(),
                      epoch + 1)

    scheduler.step(test_loss.item())

    # Print or log learning rate if needed
    current_lr = optimiser.param_groups[0]['lr']

    writer.add_scalar('lr',
                      current_lr,
                      epoch + 1)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()/batch_size} ({avg_loss.item()/len(train_loader)}), Holdout Loss: {test_loss}, LR: {current_lr}")

    writer.add_embedding(anchor_embedding, global_step=epoch)
    torch.save(siamese_model.state_dict(), f'{save_folder}/siamese_model.pth')
writer.close()


