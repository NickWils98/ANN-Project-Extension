import torch.nn as nn
import torch.optim as optim
import warnings
import torch
from ScoreCam import score_cam
from ModelFunctions import *
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image, ImageFilter

import torch
from torch.optim import SGD
from torch.autograd import Variable
from torchvision import models

from misc_functions import recreate_image, save_image
from RegularizedUnitSpecificImageGeneration import RegularizedClassSpecificImageGeneration

DIRECTORYMODEL = os.path.join("fully-supervised", "bs16_lr001_epochs10")


# Function to train the model
def train_model(model, train_loader, val_loader, num_epochs, device, optimizer, criterion):
    train_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.set_grad_enabled(True):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

            _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        accuracy_train = correct_train / total_train

        train_losses.append(epoch_loss)
        train_accuracies.append(accuracy_train)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {epoch_loss:.4f} - Training Accuracy: {accuracy_train:.4f}")

        val_accuracy = evaluate_model(model, val_loader, device)
        val_accuracies.append(val_accuracy)
        # model_save_path = os.path.join(DIRECTORYMODEL, f'model{epoch}.pth')
        # torch.save(model.state_dict(), model_save_path)
    return train_accuracies, val_accuracies, train_losses

# Function to evaluate the model
def evaluate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad(), torch.set_grad_enabled(False):
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == '__main__':

    data_dir = f'15SceneData'
    num_classes = 15
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    num_fc_layers = 0
    fc_hidden_units = 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_loader, val_loader = load_datasets_in_loaders(data_dir, batch_size)
    model = create_model(num_classes, )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # train_accuracies, val_accuracies, train_losses = train_model(model, train_loader, val_loader, num_epochs, device, optimizer, criterion)

    # write_metrics(train_accuracies, val_accuracies, train_losses, DIRECTORYMODEL)

    # plot_metrics(train_accuracies, val_accuracies, train_losses, batch_size, learning_rate, num_epochs,num_fc_layers, DIRECTORYMODEL)
    #
    # best_epoch_val = max(val_accuracies)
    # best_epoch = val_accuracies.index(best_epoch_val)
    #
    # model_path = os.path.join(DIRECTORYMODEL, f'model{best_epoch}.pth')
    # score_cam(model_path, val_loader, device, [0,100,200,300,400,500,600,700,800,900], num_classes, DIRECTORYMODEL)


    model_path = os.path.join(DIRECTORYMODEL, f'model{6}.pth')

    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Select the last convolutional layer and the corresponding SiLU activation layer
    conv_layer = model.features[8][0]
    # Get the weights of the selected convolutional layer
    conv_weights = conv_layer.weight.data

    # Sum up the weights along each filter
    filter_weights_sum = conv_weights.view(conv_weights.size(0), -1).sum(dim=1)

    indexed_list = list(enumerate(filter_weights_sum))
    sorted_indexed_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)
    top_filters_indices = [index for index, _ in sorted_indexed_list[:5]]

    newmodel = torch.nn.Sequential(*(list(model.children())[:-2]))
    for i in range(16):
        csig = RegularizedClassSpecificImageGeneration(model, i)
        csig.generate()


