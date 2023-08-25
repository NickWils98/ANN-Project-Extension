import torch.nn as nn
import torch.optim as optim
import warnings
import torch
from ScoreCam import score_cam
from ModelFunctions import *
import os
from ModelInversion import model_inversion

num_classes = 15
batch_size = 32
num_epochs = 50
learning_rate = 0.001
num_fc_layers = 0
fc_hidden_units = 256

DIRECTORYMODEL = os.path.join("fully-supervised", f"bs{batch_size}_lr{str(learning_rate)[2:]}_epochs{num_epochs}")
# DIRECTORYMODEL = os.path.join("fully-supervised", f"bs{batch_size}_lr{str(learning_rate)[2:]}_epochs{num_epochs}fc{num_fc_layers}")


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
        model_save_path = os.path.join(DIRECTORYMODEL, f'model{epoch}.pth')
        if(epoch>15):
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


if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    data_dir = f'15SceneData'


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_loader, val_loader = load_datasets_in_loaders(data_dir, batch_size)
    model = create_model(num_classes,num_fc_layers, fc_hidden_units)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # train_accuracies, val_accuracies, train_losses = train_model(model, train_loader, val_loader, num_epochs, device, optimizer, criterion)
    #
    # best_epoch_val = max(val_accuracies)
    # best_epoch = val_accuracies.index(best_epoch_val)
    #
    # write_metrics(train_accuracies, val_accuracies, train_losses, best_epoch, DIRECTORYMODEL)
    #
    # plot_metrics(train_accuracies, val_accuracies, train_losses, batch_size, learning_rate, num_epochs,num_fc_layers, DIRECTORYMODEL)

    #
    # model_path = os.path.join(DIRECTORYMODEL, f'model{19}.pth')
    # # model_path = os.path.join(DIRECTORYMODEL, f'model{4}.pth')
    # score_cam(model, model_path, val_loader, data_dir, device, [0,1,2,3,4,5,6,7,8,9], DIRECTORYMODEL)

    model_path = os.path.join(DIRECTORYMODEL, f'model{49}.pth')

    model_inversion(model, model_path, 500, "fully")