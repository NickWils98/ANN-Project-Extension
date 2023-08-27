import warnings

import torch
import torch.optim as optim

from ModelFunctions import *
from ModelInversion import model_inversion
from ScoreCam import score_cam

# Parameters
num_classes = 15
batch_size = 32
num_epochs = 10
learning_rate = 0.0005
num_fc_layers = 0
fc_hidden_units = 256

# Directory to save models and plots
DIRECTORYMODEL = os.path.join("fully-supervised", f"bs{batch_size}_lr{str(learning_rate)[2:]}_epochs{num_epochs}")
# DIRECTORYMODEL = os.path.join("fully-supervised", f"bs{batch_size}_lr{str(learning_rate)[2:]}_epochs{num_epochs}fc{num_fc_layers}")


# Function to train the model
def train_model(model, train_loader, val_loader, num_epochs, device, optimizer, criterion):
    # list for metrics, filled in each epoch
    train_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # set model to training
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Go over all the data in the training loader
        for inputs, labels in train_loader:
            # set to device for gpu
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.set_grad_enabled(True):
                optimizer.zero_grad()
                # run model on images from the batch
                outputs = model(inputs)
                # get the loss
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            # calculate metrics
            running_loss += loss.item()

            _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        accuracy_train = correct_train / total_train

        train_losses.append(epoch_loss)
        train_accuracies.append(accuracy_train)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {epoch_loss:.4f} - Training Accuracy: {accuracy_train:.4f}")
        # evaluate this epoch
        val_accuracy = evaluate_model(model, val_loader, device)
        val_accuracies.append(val_accuracy)
        # save this epoch
        model_save_path = os.path.join(DIRECTORYMODEL, f'model{epoch}.pth')
        torch.save(model.state_dict(), model_save_path)
    return train_accuracies, val_accuracies, train_losses


# Function to evaluate the model
def evaluate_model(model, val_loader, device):
    # set model to evaluation
    model.eval()
    correct = 0
    total = 0
    # make sure no parameters are altered
    with torch.no_grad(), torch.set_grad_enabled(False):
        # Go over all the data in the validation loader
        for inputs, labels in val_loader:
            # set to device for gpu
            inputs, labels = inputs.to(device), labels.to(device)
            # run model on images from the batch
            outputs = model(inputs)
            # calculate metrics
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")
    return accuracy


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # Make the subdirectory if it doesn't exist yet
    if not os.path.exists(DIRECTORYMODEL):
        os.makedirs(DIRECTORYMODEL)

    # directory where the data is stored
    data_dir = f'15SceneData'
    # set the device for gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # load the training and validation data
    train_loader, val_loader = load_datasets_in_loaders(data_dir, batch_size)
    # create the model
    model = create_model(num_classes, num_fc_layers, fc_hidden_units)
    model.to(device)

    # create loss function
    criterion = nn.CrossEntropyLoss()
    # create optimiser
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # train the model
    train_accuracies, val_accuracies, train_losses = train_model(model, train_loader, val_loader, num_epochs, device,
                                                                 optimizer, criterion)

    # get the best epoch based on validation accuracy
    best_epoch_val = max(val_accuracies)
    best_epoch = val_accuracies.index(best_epoch_val)
    # write metrics away
    write_metrics(train_accuracies, val_accuracies, train_losses, best_epoch, DIRECTORYMODEL)
    # plot the metrics
    plot_metrics(train_accuracies, val_accuracies, train_losses, batch_size, learning_rate, num_epochs, num_fc_layers,
                 DIRECTORYMODEL)

    # make scoreCAM
    model_path = os.path.join(DIRECTORYMODEL, f'model{best_epoch}.pth')
    # model_path = os.path.join(DIRECTORYMODEL, f'model{3}.pth')
    score_cam(model, model_path, val_loader, data_dir, device, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], DIRECTORYMODEL)

    # make model inversion
    model_path = os.path.join(DIRECTORYMODEL, f'model{74}.pth')
    model_inversion(model, model_path, 1500, "fully9")
