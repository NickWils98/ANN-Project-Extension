import warnings

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import rotate

from ModelFunctions import *
from ModelInversion import model_inversion
from ScoreCam import score_cam_batch


# Class for rotation
class RotationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, rotations=[0, 90, 180, 270]):
        # dataset
        self.dataset = dataset
        # get rotations
        self.rotations = rotations

    def __getitem__(self, index):
        # get original image and label
        image, label = self.dataset[index]
        # rotated the image
        rotated_images = [rotate(image, angle) for angle in self.rotations]
        # get the labels
        rotated_labels = torch.tensor([self.rotations.index(angle) for angle in self.rotations])
        return rotated_images, rotated_labels

    def __len__(self):
        return len(self.dataset)


# Parameters
num_classes = 4
batch_size = 16
num_epochs = 10
learning_rate = 0.0005
num_fc_layers = 0
fc_hidden_units = 256

# Directory to save models and plots
DIRECTORYMODEL = os.path.join("rotation", f"bs{batch_size}_lr{str(learning_rate)[2:]}_epochs{num_epochs}")
# DIRECTORYMODEL = os.path.join("rotation", f"bs{batch_size}_lr{str(learning_rate)[2:]}_epochs{num_epochs}fc{num_fc_layers}")


# Function to train the model
def train_rotation_model(mode, train_loader, val_loader, num_epochs, device, optimizer, criterion):
    to_pil = ToPILImage()
    draw = False
    rotation_angles = [0, 90, 180, 270]
    # list for metrics, filled in each epoch
    train_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # set model to training
        mode.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Go over all the data in the training loader
        for batch_rotated_images, labels in train_loader:
            all_images = []
            rotated_labels = []
            # get all the perturbated images of the batch in the same list
            for i, rotated_image_list in enumerate(batch_rotated_images):
                all_images.extend(rotated_image_list)
                rotated_labels.extend([i] * len(rotated_image_list))

            inputs = torch.stack(all_images).to(device)
            rotated_labels = torch.tensor(rotated_labels).to(device)

            with torch.set_grad_enabled(True):
                optimizer.zero_grad()
                # run model on images from the batch
                outputs = mode(inputs)
                # get the loss
                loss = criterion(outputs, rotated_labels)
                loss.backward()
                optimizer.step()
            # calculate metrics
            running_loss += loss.item()

            _, predicted_train = torch.max(outputs.data, 1)
            total_train += rotated_labels.size(0)
            correct_train += (predicted_train == rotated_labels).sum().item()

            # on the first epoch draw the perturbation
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
        # evaluate this epoch
        val_accuracy = evaluate_rotation_model(model, val_loader, device)
        val_accuracies.append(val_accuracy)
        # save this epoch
        model_save_path = os.path.join(DIRECTORYMODEL, f'model{epoch}.pth')
        torch.save(model.state_dict(), model_save_path)
    return train_accuracies, val_accuracies, train_losses


# Function to evaluate the model
def evaluate_rotation_model(model, val_loader, device):
    # set model to evaluation
    model.eval()
    correct = 0
    total = 0
    # make sure no parameters are altered
    with torch.no_grad(), torch.set_grad_enabled(False):
        for batch_rotated_images, labels in val_loader:
            all_images = []
            rotated_labels = []
            # Go over all the data in the validation loader
            for i, rotated_image_list in enumerate(batch_rotated_images):
                all_images.extend(rotated_image_list)
                rotated_labels.extend([i] * len(rotated_image_list))
            # set to device for gpu
            inputs = torch.stack(all_images).to(device)
            rotated_labels = torch.tensor(rotated_labels).to(device)
            # run model on images from the batch
            outputs = model(inputs)
            # calculate metrics
            _, predicted = torch.max(outputs.data, 1)
            total += rotated_labels.size(0)
            correct += (predicted == rotated_labels).sum().item()

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
    train_loader, val_loader = load_datasets_in_loaders(data_dir, batch_size, RotationDataset)
    # create the model
    model = create_model(num_classes, num_fc_layers, fc_hidden_units)
    model.to(device)

    # create loss function
    criterion = nn.CrossEntropyLoss()
    # create optimiser
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # train the model
    train_accuracies, val_accuracies, train_losses = train_rotation_model(model, train_loader, val_loader, num_epochs,
                                                                          device, optimizer, criterion)

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
    # model_path = os.path.join(DIRECTORYMODEL, f'model{9}.pth')
    score_cam_batch(model, model_path, val_loader, device, [0, 1, 2, 3, 4], DIRECTORYMODEL, 1)

    # make model inversion
    model_path = os.path.join(DIRECTORYMODEL, f'model{9}.pth')
    model_inversion(model, model_path, 1500, "rotation39")
