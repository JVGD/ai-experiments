import numpy as np
import torch
from torch.nn import Module, Linear
from torch.nn.functional import relu
from tqdm import tqdm

# Sanity check
if torch.__version__ != '1.4.0':
    raise ValueError('This must be run with pytorch 1.4.0')


class Model(Module):
    """Model creation: subclassing approach"""
    def __init__(self):
        super().__init__()

        # Layers instantiation (no need for Input layer)
        self.linear1 = Linear(in_features=(20), out_features=(20))
        self.linear2 = Linear(in_features=(20), out_features=(10))
        self.linear3 = Linear(in_features=(10), out_features=(1))

    def forward(self, x):
        x = self.linear1(x)
        x = relu(x)
        x = self.linear2(x)
        x = relu(x)
        x = self.linear3(x)
        return x

    # NOTE no need to implement backward
    # since it is already infered


def loss_compute(y_true, y_pred):
    return (y_true - y_pred)**2


def train():
    # Learn to sum 20 nums
    train_samples = torch.randn(size=(10000, 20))
    train_targets = torch.sum(train_samples, dim=-1)
    test_samples = torch.randn(size=(100, 20))
    test_targets = torch.sum(test_samples, dim=-1)

    # Model
    model = Model()

    # Training loop
    epochs = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):

        # Fancy progress bar
        pbar = tqdm(range(len(train_samples)))

        # Metrics
        loss_metric = []

        # Batches iteration, batch_size = 1
        for batch_id in pbar:

            # Getting sample target pair
            sample = train_samples[batch_id]
            target = train_targets[batch_id]

            # Adding batch dim since batch=1
            sample = sample.unsqueeze(0)
            target = target.unsqueeze(0)

            # Forward pass: needs to be recorded by gradient tape
            target_pred = model(sample)
            loss = loss_compute(target, target_pred)

            # Backward pass:             
            # Init previous gradients to 0
            optimizer.zero_grad()
            # Running backward pass for computing gradients
            loss.backward()
            # Update weights
            optimizer.step()

            # Tracking progress
            loss_metric.append(loss.item())
            loss_metric_avg = sum(loss_metric) / (batch_id+1)
            pbar.set_description('Training Loss:   %.3f' % loss_metric_avg)

        # At the end of the epoch test the model
        test_targets_pred = model(test_samples)
        test_targets_pred = test_targets_pred.squeeze()
        test_loss = loss_compute(test_targets, test_targets_pred)
        test_loss_avg = torch.mean(test_loss).item()
        print('Validation Loss: %.3f' % test_loss_avg)


if __name__ == '__main__':
    train()