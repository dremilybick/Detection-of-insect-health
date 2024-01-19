import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self, input_size=400, embedding_size=64):
        super(SiameseNetwork, self).__init__()

        self.conv1 = nn.Conv1d(1,32,kernel_size=5)
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=5)
        self.pool3 = nn.MaxPool1d(2)
        self.dropout3 = nn.Dropout(0.25)

        self.flatten = nn.Flatten()

        fc_input_size = self.calculate_fc_input_size()
        self.fc1 = nn.Linear(fc_input_size, 128)
        nn.init.xavier_uniform_(self.fc1.weight)

        self.dropout4 = nn.Dropout(0.5)

        self.embedding = nn.Linear(128, embedding_size)
        nn.init.xavier_uniform_(self.embedding.weight)

    def calculate_fc_input_size(self):
        # Dummy input to calculate the output size of the convolutional layers
        dummy_input = torch.randn(1, 1, 400)
        x = self.conv1(dummy_input)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        x = self.flatten(x)
        return x.numel()

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)

        output_embedding = torch.sigmoid(self.embedding(x))

        return output_embedding


class SiameseSpectral(nn.Module):
    def __init__(self, embedding_size, dropout_prob=0.5, batch_size=16):
        super(SiameseSpectral, self).__init__()

        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.dropout = nn.Dropout(p=dropout_prob)

        fc_input_size = self.calculate_fc_input_size()
        self.fc1 = nn.Linear(fc_input_size, 256)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(256, 128)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.embedding = nn.Linear(128, embedding_size)
        nn.init.xavier_uniform_(self.embedding.weight)


    def calculate_fc_input_size(self):
        # Dummy input to calculate the output size of the convolutional layers
        dummy_input = torch.randn(1, 1, 129,15)
        x = self.conv1(dummy_input)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)

        return x.numel()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)  # input size (129, 15)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.embedding(x)
        return x

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor_embedding, positive_embedding, negative_embedding):
        # Calculate pairwise distances
        positive_distance = F.pairwise_distance(anchor_embedding, positive_embedding)
        negative_distance = F.pairwise_distance(anchor_embedding, negative_embedding)

        # Calculate contrastive loss
        loss = torch.mean(F.relu(self.margin + positive_distance - negative_distance))

        return loss