import torch
import torch.optim as optim

from ModelFunctions import *
from ModelInversion import model_inversion
from ScoreCam import score_cam

# Parameters pretext model
old_num_classes = 4
old_batch_size = 16
old_learning_rate = 0.001
old_num_fc_layers = 0
old_fc_hidden_units = 256
old_epoch_num = 2

# Directory to save models and plots old Rotation
OLDDIRECTORYMODEL = os.path.join("rotation", f"bs{old_batch_size}_lr{str(old_learning_rate)[2:]}_epochs{10}")
# OLDDIRECTORYMODEL = os.path.join("rotation", f"bs{old_batch_size}_lr{str(old_learning_rate)[2:]}_epochs{10}fc{old_num_fc_layers}")

# Directory to save models and plots old Perturbation
# OLDDIRECTORYMODEL = os.path.join("perturbation", f"bs{old_batch_size}_lr{str(old_learning_rate)[2:]}_epochs{10}")
# OLDDIRECTORYMODEL = os.path.join("perturbation", f"bs{old_batch_size}_lr{str(old_learning_rate)[2:]}_epochs{10}fc{old_num_fc_layers}")



# Parameters scene classifier
num_classes = 15
batch_size = 32
num_epochs = 10
learning_rate = 0.001
num_fc_layers = 0
fc_hidden_units = 256

# Directory to save models and plots Rotation
DIRECTORYMODEL = os.path.join("scene_rotation", f"bs{batch_size}_lr{str(learning_rate)[2:]}_epochs{10}")
# DIRECTORYMODEL = os.path.join("scene_rotation", f"bs{batch_size}_lr{str(learning_rate)[2:]}_epochs{3}fc{num_fc_layers}")

# Directory to save models and plots Perturbation
# DIRECTORYMODEL = os.path.join("scene_perturbation", f"bs{batch_size}_lr{str(learning_rate)[2:]}_epochs{3}")
# DIRECTORYMODEL = os.path.join("scene_perturbation", f"bs{batch_size}_lr{str(learning_rate)[2:]}_epochs{3}fc{num_fc_layers}")


# Function to load and configure the model right model
def load_model(model, model_path, num_fc_layers=0, fc_hidden_units=256):
    # load the right weights in the model
    model.load_state_dict(torch.load(model_path))
    # get the amount of filters for the input of the first layer of the classifier
    if old_num_fc_layers == 0:
        num_ftrs = model.classifier[1].in_features
    else:
        num_ftrs = model.classifier[1][0].in_features

    # if there are no extra layers needed just create 1 FC
    if num_fc_layers == 0:
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    else:
        # Create a sequential container to stack multiple fully-connected layers
        fc_layers = []
        for _ in range(num_fc_layers):
            fc_layers.append(nn.Linear(num_ftrs, fc_hidden_units))
            fc_layers.append(nn.ReLU(inplace=True))
            num_ftrs = fc_hidden_units

        # end with a final layer to the classes count
        fc_layers.append(nn.Linear(fc_hidden_units, num_classes))
        model.classifier[1] = nn.Sequential(*fc_layers)

    # Set all layers to be non-trainable
    for param in model.parameters():
        param.requires_grad = False

    # Set the parameters for the classifier to trainable
    for m in model.classifier[0].parameters():
        m.requires_grad = True
    for m in model.classifier[1].parameters():
        m.requires_grad = True

    return model


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
    # load the old pretext model
    model_path_old = os.path.join(OLDDIRECTORYMODEL, f'model{old_epoch_num}.pth')
    # create the model
    model = create_model(old_num_classes, old_num_fc_layers, old_fc_hidden_units)
    model = load_model(model, model_path_old)
    # create the model
    model.to(device)

    # create loss function
    criterion = nn.CrossEntropyLoss()
    # create optimiser
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

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
    # model_path = os.path.join(DIRECTORYMODEL, f'model{best_epoch}.pth')
    model_path = os.path.join(DIRECTORYMODEL, f'model{9}.pth')
    score_cam(model, model_path, val_loader, data_dir, device, [13,907,1004,1205,1606, 1806,2500,2700], DIRECTORYMODEL)

    # # make model inversion
    model_path = os.path.join(DIRECTORYMODEL, f'model{9}.pth')
    model_inversion(model, model_path, 150, "scene_rotation")
