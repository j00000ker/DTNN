"""Training utilities."""

import numpy as np
import torch
from datetime import datetime


def train_model(model, criterion, optimizer, train_loader, val_loader, epochs, model_path, device):
    """Standard training loop with validation and model checkpointing.

    Parameters
    ----------
    model : torch.nn.Module
    criterion : loss function
    optimizer : torch.optim.Optimizer
    train_loader : DataLoader
    val_loader : DataLoader
    epochs : int
    model_path : str
        File path to save the best model (uses ``state_dict``).
    device : torch.device

    Returns
    -------
    train_losses, val_losses : np.ndarray
    """
    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)
    best_val_loss = np.inf
    best_epoch = 0

    for it in range(epochs):
        model.train()
        t0 = datetime.now()

        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.int64)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        for inputs, targets in val_loader:
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.int64)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
        val_loss /= len(val_loader)

        train_losses[it] = train_loss
        val_losses[it] = val_loss

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), model_path)
            best_val_loss = val_loss
            best_epoch = it
            print('model saved')

        dt = datetime.now() - t0
        print(
            f'Epoch {it + 1}/{epochs}, Train Loss: {train_loss:.4f}, '
            f'Validation Loss: {val_loss:.4f}, Duration: {dt}, Best Val Epoch: {best_epoch + 1}'
        )

    return train_losses, val_losses


def evaluate(model, test_loader, device):
    """Evaluate a trained model on a test set.

    Returns accuracy score, list of all targets, and list of all predictions.
    """
    all_targets = []
    all_predictions = []
    n_correct = 0
    n_total = 0

    model.eval()
    for inputs, targets in test_loader:
        inputs = inputs.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.int64)

        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)

        n_correct += (predictions == targets).sum().item()
        n_total += targets.shape[0]

        all_targets.append(targets.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())

    test_acc = n_correct / n_total
    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)

    return test_acc, all_targets, all_predictions
