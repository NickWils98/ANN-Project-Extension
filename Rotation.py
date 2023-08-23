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
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms.functional import rotate
import os
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import warnings
from ScoreCam import score_cam_batch
from ModelFunctions import *
import os

class RotationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, rotations=[0, 90, 180, 270]):
        self.dataset = dataset
        self.rotations = rotations

    def __getitem__(self, index):
        image, label = self.dataset[index]
        rotated_images = [rotate(image, angle) for angle in self.rotations]
        rotated_labels = torch.tensor([self.rotations.index(angle) for angle in self.rotations])
        return rotated_images, rotated_labels


    def __len__(self):
        return len(self.dataset)



num_classes = 4
batch_size = 16
num_epochs = 10
learning_rate = 0.001
num_fc_layers = 1
fc_hidden_units = 256

# DIRECTORYMODEL = os.path.join("rotation", f"bs{batch_size}_lr{str(learning_rate)[2:]}_epochs{num_epochs}")
DIRECTORYMODEL = os.path.join("rotation", f"bs{batch_size}_lr{str(learning_rate)[2:]}_epochs{num_epochs}fc{num_fc_layers}")



# Function to train the model
def train_rotation_model(mode, train_loader, val_loader, num_epochs, device, optimizer, criterion):
    to_pil = ToPILImage()
    draw = False
    rotation_angles = [0, 90, 180, 270]
    train_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        mode.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_rotated_images, labels in train_loader:
            all_images = []
            rotated_labels = []

            for i, rotated_image_list in enumerate(batch_rotated_images):
                all_images.extend(rotated_image_list)
                rotated_labels.extend([i] * len(rotated_image_list))

            inputs = torch.stack(all_images).to(device)
            rotated_labels = torch.tensor(rotated_labels).to(device)


            with torch.set_grad_enabled(True):
                optimizer.zero_grad()
                outputs = mode(inputs)
                loss = criterion(outputs, rotated_labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

            _, predicted_train = torch.max(outputs.data, 1)
            total_train += rotated_labels.size(0)
            correct_train += (predicted_train == rotated_labels).sum().item()

            if draw == False:  # Plot only in the first epoch
                draw = True
                plt.figure(figsize=(6, 3))
                for i, rotated_image in enumerate(batch_rotated_images):
                    plt.subplot(1, len(rotation_angles), i + 1)
                    plt.title(f"Rotation {rotation_angles[i]}°")
                    plt.imshow(to_pil(rotated_image[0]))
                    plt.axis("off")
                plt.tight_layout()
                plot_save_counter = os.path.join(DIRECTORYMODEL, f'rotation.png')
                plt.savefig(plot_save_counter, dpi=300)
                plt.show()

        epoch_loss = running_loss / len(train_loader)
        accuracy_train = correct_train / total_train

        train_losses.append(epoch_loss)
        train_accuracies.append(accuracy_train)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {epoch_loss:.4f} - Training Accuracy: {accuracy_train:.4f}")

        val_accuracy = evaluate_rotation_model(model, val_loader, device)
        val_accuracies.append(val_accuracy)

        model_save_path = os.path.join(DIRECTORYMODEL, f'model{epoch}.pth')
        torch.save(model.state_dict(), model_save_path)
    return train_accuracies, val_accuracies, train_losses


# Function to evaluate the model
def evaluate_rotation_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad(), torch.set_grad_enabled(False):
        for batch_rotated_images, labels in val_loader:
            all_images = []
            rotated_labels = []

            for i, rotated_image_list in enumerate(batch_rotated_images):
                all_images.extend(rotated_image_list)
                rotated_labels.extend([i] * len(rotated_image_list))

            inputs = torch.stack(all_images).to(device)
            rotated_labels = torch.tensor(rotated_labels).to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += rotated_labels.size(0)
            correct += (predicted == rotated_labels).sum().item()

    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")
    return accuracy


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    data_dir = f'15SceneData'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_loader, val_loader = load_datasets_in_loaders(data_dir, batch_size, RotationDataset)
    model = create_model(num_classes, num_fc_layers, fc_hidden_units)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_accuracies, val_accuracies, train_losses = train_rotation_model(model, train_loader, val_loader, num_epochs, device, optimizer, criterion)

    best_epoch_val = max(val_accuracies)
    best_epoch = val_accuracies.index(best_epoch_val)

    write_metrics(train_accuracies, val_accuracies, train_losses, best_epoch, DIRECTORYMODEL)

    plot_metrics(train_accuracies, val_accuracies, train_losses, batch_size, learning_rate, num_epochs, num_fc_layers,
                 DIRECTORYMODEL)

    model_path = os.path.join(DIRECTORYMODEL, f'model{best_epoch}.pth')
    # model_path = os.path.join(DIRECTORYMODEL, f'model{4}.pth')
    score_cam_batch(model, model_path, val_loader, device, [0,1,2,3,4], DIRECTORYMODEL, 1)