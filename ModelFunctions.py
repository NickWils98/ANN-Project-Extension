import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchcam.utils import overlay_mask
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
from torchcam.methods import ScoreCAM
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import warnings
import numpy as np

import torch
from torchvision import transforms
from torch.optim import SGD
import numpy as np
from PIL import Image
import os

# Function to load datasets and data loaders
def load_datasets_in_loaders(data_dir, batch_size, transform_class=None):

    transform = models.EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()

    train_dataset = ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    if transform_class is not None:
        train_dataset = transform_class(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = ImageFolder(os.path.join(data_dir, "validation"), transform=transform)
    if transform_class is not None:
        val_dataset = transform_class(val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# Function to create and configure the model
def create_model(num_classes,  num_fc_layers=0, fc_hidden_units=256):
    model = models.efficientnet_b0(pretrained='imagenet')
    num_ftrs = model.classifier[1].in_features

    if num_fc_layers == 0:
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    else:
        # Create a sequential container to stack multiple fully-connected layers
        fc_layers = []
        for _ in range(num_fc_layers):
            fc_layers.append(nn.Linear(num_ftrs, fc_hidden_units))
            fc_layers.append(nn.ReLU(inplace=True))
            num_ftrs = fc_hidden_units

        fc_layers.append(nn.Linear(fc_hidden_units, num_classes))
        model.classifier[1] = nn.Sequential(*fc_layers)

    for param in model.parameters():
        param.requires_grad = True

    return model

def write_metrics(train_accuracies, val_accuracies,train_losses, best_epoch, plot_path):
    epochs = len(val_accuracies)
    with open(os.path.join(plot_path, 'training_metrics.txt'), 'w') as file:
        file.write(f"Best Epoch for Validation Accuracy = {best_epoch}\n")
        file.write("\nEpoch\tValidation Accuracy\n")
        for epoch in range(epochs):
            file.write(
                f"{epoch + 1}\t{val_accuracies[epoch]:.4f}\n")
        file.write("\nEpoch\tTraining Accuracy\n")
        for epoch in range(epochs):
            file.write(
                f"{epoch + 1}\t{train_accuracies[epoch]:.4f}\n")
        file.write("\nEpoch\tLoss\n")
        for epoch in range(epochs):
            file.write(
                f"{epoch + 1}\t{train_losses[epoch]:.4f}\n")


def plot_metrics(train_accuracies, val_accuracies,train_losses, batch_size, learning_rate, num_epochs, num_fc_layers, plot_path):

    epochs = np.arange(1, len(train_accuracies) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(epochs, train_losses, label='Training Loss', color='tab:red', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:blue')
    ax2.plot(epochs, train_accuracies, label='Training Accuracy', color='tab:blue', linestyle='dashed', marker='o')
    ax2.plot(epochs, val_accuracies, label='Validation Accuracy', color='tab:orange', linestyle='dashed', marker='o')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # fig.tight_layout()
    plt.title(
        f'Metrics During Training\nBatch Size: {batch_size}, Learning Rate: {learning_rate}, Epochs: {num_epochs}',
        fontsize=16)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True)

    # Save the plot as a PNG image
    plot_path = os.path.join(plot_path, f'metrics_plot_bs{batch_size}_lr{str(learning_rate)[2:]}_epochs{num_epochs+1}_fc{num_fc_layers}.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Metrics plot saved as {plot_path}")
    plt.show()