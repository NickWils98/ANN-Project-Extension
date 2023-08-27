import random

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from torchvision.transforms import ToTensor

from ModelFunctions import *
from ModelInversion import model_inversion
from ScoreCam import score_cam_batch


# Class for perturbation
class PerturbationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        # dataset
        self.dataset = dataset
        # shape of the perturbation
        self.perturbation_shape = (10, 10)
        # functions to transform image
        self.to_pil = ToPILImage()
        self.to_tensor = ToTensor()

    def apply_perturbation(self, image):
        # Convert the tensor back to PIL image
        image = self.to_pil(image)
        # create copies for black and white perturbation
        perturbed_image_black = image.copy()
        perturbed_image_white = image.copy()

        h, w = perturbed_image_black.size

        # Generate random positions for the black square
        top_black = random.randint(0, h - self.perturbation_shape[0])
        left_black = random.randint(0, w - self.perturbation_shape[1])
        perturbed_image_black.paste(0, (
            left_black, top_black, left_black + self.perturbation_shape[1], top_black + self.perturbation_shape[0]))

        # Generate random positions for the white square
        top_white = random.randint(0, h - self.perturbation_shape[0])
        left_white = random.randint(0, w - self.perturbation_shape[1])
        perturbed_image_white.paste((255, 255, 255), (
            left_white, top_white, left_white + self.perturbation_shape[1], top_white + self.perturbation_shape[0]))

        return perturbed_image_black, perturbed_image_white

    def __getitem__(self, index):
        # get original image and label
        original_image, label = self.dataset[index]
        # create altered images
        perturbed_image_black, perturbed_image_white = self.apply_perturbation(original_image)
        # convert images back to tensors
        perturbed_image_black = self.to_tensor(perturbed_image_black)
        perturbed_image_white = self.to_tensor(perturbed_image_white)
        # set images and labels
        images = [perturbed_image_black, perturbed_image_white]
        labels = torch.tensor([0, 1])
        return images, labels

    def __len__(self):
        return len(self.dataset)


# Parameters
num_classes = 2
batch_size = 16
num_epochs = 10
learning_rate = 0.001
num_fc_layers = 0
fc_hidden_units = 256

# Directory to save models and plots
DIRECTORYMODEL = os.path.join("perturbation", f"bs{batch_size}_lr{str(learning_rate)[2:]}_epochs{num_epochs}")
# DIRECTORYMODEL = os.path.join("perturbation", f"bs{batch_size}_lr{str(learning_rate)[2:]}_epochs{num_epochs}fc{num_fc_layers}")


# Function to train the model
def train_rotation_model(model, train_loader, val_loader, num_epochs, device, optimizer, criterion):
    to_pil = ToPILImage()
    draw = False
    perturbation_angles = [0, 1]
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
        for batch_images, labels in train_loader:
            all_images = []
            perturbated_labels = []
            # get all the perturbated images of the batch in the same list
            for i, perturbated_image_list in enumerate(batch_images):
                all_images.extend(perturbated_image_list)
                perturbated_labels.extend([i] * len(perturbated_image_list))

            inputs = torch.stack(all_images).to(device)
            perturbated_labels = torch.tensor(perturbated_labels).to(device)

            with torch.set_grad_enabled(True):
                optimizer.zero_grad()
                # run model on images from the batch
                outputs = model(inputs)
                # get the loss
                loss = criterion(outputs, perturbated_labels)
                loss.backward()
                optimizer.step()
            # calculate metrics
            running_loss += loss.item()

            _, predicted_train = torch.max(outputs.data, 1)
            total_train += perturbated_labels.size(0)
            correct_train += (predicted_train == perturbated_labels).sum().item()

            # on the first epoch draw the pertubation
            if draw == False:  # Plot only in the first epoch
                draw = True
                plt.figure(figsize=(6, 3))
                for i, perturbated_image in enumerate(batch_images):
                    plt.subplot(1, len(perturbation_angles), i + 1)
                    plt.title(f"Perturbation  {perturbation_angles[i]}°")
                    plt.imshow(to_pil(perturbated_image[0]))
                    plt.axis("off")
                plt.tight_layout()
                plot_save_counter = os.path.join(DIRECTORYMODEL, f'perturbation.png')
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
    # Make the subdirectory if it doesn't exist yet
    if not os.path.exists(DIRECTORYMODEL):
        os.makedirs(DIRECTORYMODEL)

    # directory where the data is stored
    data_dir = f'15SceneData'
    # set the device for gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # load the training and validation data
    train_loader, val_loader = load_datasets_in_loaders(data_dir, batch_size, PerturbationDataset)
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
    # model_path = os.path.join(DIRECTORYMODEL, f'model{best_epoch}.pth')
    model_path = os.path.join(DIRECTORYMODEL, f'model{2}.pth')
    score_cam_batch(model, model_path, val_loader, device, [1004], DIRECTORYMODEL, 0)

    # make model inversion
    model_path = os.path.join(DIRECTORYMODEL, f'model{9}.pth')
    model_inversion(model, model_path, 1500, "perturbation3")
