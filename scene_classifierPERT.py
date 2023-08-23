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
# Create a custom transformation for rotation classification

import warnings

# Function to load datasets and data loaders
def load_datasets_and_loaders(data_dir, batch_size):
    # transform = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    transform = models.EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()

    train_dataset = ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = ImageFolder(os.path.join(data_dir, "validation"), transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, val_dataset

# Function to create and configure the model
def load_model(model_path, num_classes):
    model = models.efficientnet_b0(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path))
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    # Set all layers to be non-trainable
    for param in model.parameters():
        param.requires_grad = False
    for m in model.classifier[1].parameters():
        m.requires_grad = True


    return model


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

        model_save_path = os.path.join('sceneClassPERT', f'scenePert{epoch}.pth')
        torch.save(model.state_dict(), model_save_path)
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


def score_cam(model_path, val_loader, device, image_list, num_classes):
    # Load pre-trained EfficientNet-B0 model
    model = models.efficientnet_b0(pretrained='imagenet')
    counter = 0
    # Modify the classifier for rotation classification (4 classes)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    for batch_rotated_images, labels in val_loader:
        labelcounter = 0
        for img in batch_rotated_images:
            if counter in image_list:
                image_list.remove(counter)
                scorecam = ScoreCAM(model, target_layer=model.features[8])
                # Choose an image from the validation dataset for visualization

                image = img.unsqueeze(0).to(device)

                # Get the class index with the highest probability
                with torch.no_grad():
                    outputs = model(image)
                    x, predicted_class = torch.max(outputs.data, 1)
                print(f'index: {counter}, true label: {labels[labelcounter]}, guess: {predicted_class.item()}')

                # Forward the image through the model to hook into the convolutional features
                # Generate the heatmap using Score-CAM

                heatmap = scorecam(predicted_class.item(), predicted_class)  # Use .item() to get the class index as a scalar
                # Overlayed on the image
                for name, cam in zip(scorecam.target_names, heatmap):
                    result = overlay_mask(to_pil_image(img), to_pil_image(cam.squeeze(0), mode='F'), alpha=0.5)
                    plt.imshow(result)
                    plt.axis('off')
                    plt.title(f'{counter}')
                    plot_save_path = f'pertubation_full_{counter}.png'
                    plt.savefig(plot_save_path, dpi=300)
                    plt.show()
                # Once you're finished, clear the hooks on your model
                scorecam.remove_hooks()
                if len(image_list)==0:
                    return
            counter += 1
            labelcounter += 1


def plot_metrics(train_accuracies, val_accuracies,train_losses, batch_size, learning_rate, num_epochs):
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
    plot_save_path = f'metrics_plot_bs{batch_size}_lr{learning_rate}_epochs{num_epochs}.png'
    plt.savefig(plot_save_path, dpi=300)
    print(f"Metrics plot saved as {plot_save_path}")
    plt.show()


def plot_accuracies(train_accuracies, val_accuracies, train_losses, batch_size, learning_rate, num_epochs):
    epochs = np.arange(1, len(train_accuracies) + 1)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.plot(epochs, train_losses, label='Training Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Training and Validation Accuracies\nBatch Size: {batch_size}, Learning Rate: {learning_rate}, Epochs: {num_epochs}')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    data_dir = f'.\\15SceneData\\'
    num_classes = 15

    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, val_dataset = load_datasets_and_loaders(data_dir, batch_size)
    rotation_model_path = os.path.join('perturbation', f'pertubation_model{9}.pth')
    model = load_model(rotation_model_path, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    train_accuracies, val_accuracies, train_losses = train_model(model, train_loader, val_loader, num_epochs, device, optimizer, criterion)
    plot_metrics(train_accuracies, val_accuracies,train_losses, batch_size, learning_rate, num_epochs)

    model_path = os.path.join('sceneClassPERT', f'scenePert{9}.pth')
    score_cam(model_path, val_loader, device, [0,1,2,3,4,5,6,7,8,9], num_classes)
    # plot_accuracies(train_accuracies, val_accuracies,train_losses, batch_size, learning_rate, num_epochs)

