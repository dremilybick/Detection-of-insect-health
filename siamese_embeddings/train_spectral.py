import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SiameseSpectral, ContrastiveLoss
from dataloader import SiameseSpectralDataset, ContrastiveSampler
from torch.utils.tensorboard import SummaryWriter
from utils import load_data
import os

run_id = 0
while os.path.exists(f'runs/siamesespectral_run_{run_id}'):
    run_id += 1
save_folder = f'runs/siamesespectral_run_{run_id}'

writer = SummaryWriter(save_folder)

# Initialize your Siamese network and other training components
embedding_size = 64
batch_size = 16
initial_lr = 0.0001
num_epochs = 50
scheduler_patience = 3
loss_margin = 1
input_size = (129,15)

#checkpoint = f'runs/siamesespectral_run_/siamese_model.pth'
checkpoint = None

siamese_model = SiameseSpectral(embedding_size)

if checkpoint is not None:
    checkpoint = torch.load(checkpoint)
    siamese_model.load_state_dict(checkpoint)

contrastive_loss = ContrastiveLoss(margin=loss_margin)
optimiser = optim.Adam(siamese_model.parameters(), lr=initial_lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=scheduler_patience)

train_signals, train_thresholds, train_origin = load_data('train.txt')
test_signals, test_thresholds, test_origin = load_data('test.txt')

train_dataset = SiameseSpectralDataset(train_signals, train_thresholds, train_origin)
test_dataset = SiameseSpectralDataset(test_signals, test_thresholds, test_origin)

contrastive_train = ContrastiveSampler(train_dataset, batch_size=batch_size)
train_loader = DataLoader(train_dataset, batch_sampler=contrastive_train)
contrastive_test = ContrastiveSampler(train_dataset, batch_size=batch_size)
test_loader = DataLoader(train_dataset, batch_sampler=contrastive_train)

for epoch in range(num_epochs):
    siamese_model.train()
    avg_loss = 0.0
    for batch in train_loader:
        anchor, positive, negative = batch

        batch_size = len(anchor)
        anchor = anchor.reshape(batch_size,1, input_size[0], input_size[1])
        positive = positive.reshape(batch_size,1, input_size[0],input_size[1])
        negative = negative.reshape(batch_size, 1,input_size[0],input_size[1])

        # Forward pass
        anchor_embedding = siamese_model(anchor)
        positive_embedding = siamese_model(positive)
        negative_embedding = siamese_model(negative)

        # Calculate contrastive loss
        loss = contrastive_loss(anchor_embedding, positive_embedding, negative_embedding)
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
            anchor = anchor.reshape(batch_size, 1, input_size[0], input_size[1])
            positive = positive.reshape(batch_size, 1, input_size[0], input_size[1])
            negative = negative.reshape(batch_size, 1, input_size[0], input_size[1])

            # Forward pass
            anchor_embedding = siamese_model(anchor)
            positive_embedding = siamese_model(positive)
            negative_embedding = siamese_model(negative)
            test_loss += contrastive_loss(anchor_embedding, positive_embedding, negative_embedding)

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
