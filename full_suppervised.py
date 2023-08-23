import torch.nn as nn
import torch.optim as optim
import warnings
import torch
from ScoreCam import score_cam
from ModelFunctions import *
import os

num_classes = 15
batch_size = 32
num_epochs = 10
learning_rate = 0.0005
num_fc_layers = 2
fc_hidden_units = 256

# DIRECTORYMODEL = os.path.join("fully-supervised", f"bs{batch_size}_lr{str(learning_rate)[2:]}_epochs{num_epochs}")
DIRECTORYMODEL = os.path.join("fully-supervised", f"bs{batch_size}_lr{str(learning_rate)[2:]}_epochs{num_epochs}fc{num_fc_layers}")


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

    train_accuracies, val_accuracies, train_losses = train_model(model, train_loader, val_loader, num_epochs, device, optimizer, criterion)

    best_epoch_val = max(val_accuracies)
    best_epoch = val_accuracies.index(best_epoch_val)

    write_metrics(train_accuracies, val_accuracies, train_losses, best_epoch, DIRECTORYMODEL)

    plot_metrics(train_accuracies, val_accuracies, train_losses, batch_size, learning_rate, num_epochs,num_fc_layers, DIRECTORYMODEL)


    model_path = os.path.join(DIRECTORYMODEL, f'model{best_epoch}.pth')
    # model_path = os.path.join(DIRECTORYMODEL, f'model{4}.pth')
    score_cam(model, model_path, val_loader, data_dir, device, [0,1,2,3,4,5,6,7,8,9], DIRECTORYMODEL)

    # # For each model, repeat the following steps
    # model_name ='supervised'
    # print(f"Running model inversion for {model_name} model")
    #
    # # Select five filters with highest weights for this model
    # silu_layer = model.features[8][2]
    # filter_sums = []
    # for param in silu_layer.parameters():
    #     filter_sums.append(param.data.sum().item())
    #
    # # Get indices of filters with highest weights
    # num_filters_to_select = 5
    # selected_filter_indices = sorted(range(len(filter_sums)), key=lambda i: filter_sums[i], reverse=True)[
    #                           :num_filters_to_select]
    #
    # # Define the optimization parameters
    # learning_rate = 0.01
    # num_iterations = 500
    #
    # # Create an input tensor (initially random)
    # input_image = nn.Parameter(torch.randn(1, 3, 224, 224), requires_grad=True)
    # model.to('cpu')
    # optimizer = torch.optim.Adam([input_image], lr=learning_rate)
    #
    # for iteration in range(num_iterations):
    #     optimizer.zero_grad()
    #
    #     # Forward pass through the model
    #     output = model(input_image)
    #
    #     # Get the activations for the selected filters
    #     selected_activations = [output[0, idx].sum() for idx in selected_filter_indices]
    #
    #     # Compute the loss as the negative sum of selected activations
    #     loss = -sum(selected_activations)
    #     loss_tensor = torch.tensor(loss, dtype=torch.float32, requires_grad=True)
    #
    #     loss_tensor.backward()
    #     optimizer.step()
    #
    #     if iteration % 100 == 0:
    #         print(f"Iteration: {iteration}, Loss: {loss_tensor.item()}")
    #
    # # Perform model inversion
    # inverted_image = model_inversion(model, selected_filter_indices)
    #
    # # Save the synthetic image
    # synthetic_image_path = f'synthetic_image_{model_name}.png'
    # Image.fromarray((inverted_image * 255).astype('uint8')).save(synthetic_image_path)
    # print(f"Synthetic image for {model_name} model saved at {synthetic_image_path}")

    # import torch.optim as optim
    # import matplotlib.pyplot as plt
    # from torchvision import transforms
    # import torchvision.models as models
    #
    # # Load the EfficientNet model
    # # model = models.efficientnet_b0(pretrained=True)
    # model.eval()
    #
    # # Select the desired layer for model inversion (SiLu activation layer)
    # conv_layer  = model.features[8][1]  # Accessing the SiLU layer
    # image = None
    # batch_rotated_images, labels = next(iter(val_loader))
    # image = batch_rotated_images[0]
    # img = image.unsqueeze(0).to(device)
    # # Find the filter with the highest summation of weights in the selected layer
    # max_weight_sum = 0
    # selected_filter = 0
    # for filter_idx, filter_weights in enumerate(conv_layer.weight):
    #     weight_sum = filter_weights.sum()
    #     if weight_sum > max_weight_sum:
    #         max_weight_sum = weight_sum
    #         selected_filter = filter_idx
    #
    # # Model inversion settings
    # sample_index = 0
    # sample_image = image
    # sample_image = img
    # sample_image.to(device)
    #
    # # Preprocess the sample image to match training data normalization
    #
    # # Define the optimizer for the inversion process
    # optimizer = optim.SGD([sample_image.requires_grad_()], lr=0.01, momentum=0.9)
    #
    # # Run model inversion for the selected filter
    # num_iterations = 100  # Number of optimization iterations
    #
    # for _ in range(num_iterations):
    #     optimizer.zero_grad()
    #     output = conv_layer(sample_image)
    #     filter_activation = output[0, selected_filter].sum()
    #     loss = -filter_activation  # Inversion seeks to maximize the filter's activation
    #     loss.backward()
    #     optimizer.step()
    #
    # # Convert the inverted image to a numpy array
    # inverted_image = transforms.ToPILImage()(sample_image[0].cpu().detach())
    #
    # # Display the inverted image
    # plt.imshow(inverted_image)
    # plt.axis('off')
    # plt.title(f'Inverted Image for Filter {selected_filter}')
    # plt.show()
    # model_path = os.path.join(DIRECTORYMODEL, f'model{6}.pth')
    # #
    # model.load_state_dict(torch.load(model_path), strict=True)
    # model.eval()
    # model.to('cpu')
    #
    # import torch
    # import torch.nn as nn
    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    # # Select the last convolutional layer and the corresponding SiLU activation layer
    # conv_layer = model.features[8][1]
    # # Get the weights of the selected convolutional layer
    # conv_weights = conv_layer.weight.data
    #
    # # Sum up the weights along each filter
    # filter_weights_sum = conv_weights.view(conv_weights.size(0), -1).sum(dim=1)
    #
    # indexed_list = list(enumerate(filter_weights_sum))  # Create a list of (index, value) pairs
    # sorted_indexed_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)  # Sort by value in descending order
    # top_filters_indices = [index for index, _ in sorted_indexed_list[:5]]  # Get the indexes of the first 5 elements
    #
    # class Identity(nn.Module):
    #     def __init__(self):
    #         super(Identity, self).__init__()
    #
    #     def forward(self, x):
    #         return x
    # import os
    # import numpy as np
    # from PIL import Image, ImageFilter
    #
    # import torch
    # from torch.optim import SGD
    # from torch.autograd import Variable
    # from torchvision import models
    #
    # from misc_functions import recreate_image, save_image
    # from RegularizedUnitSpecificImageGeneration import RegularizedClassSpecificImageGeneration
    #
    # target_class = 0  # Flamingo
    # newmodel = torch.nn.Sequential(*(list(model.children())[:-2]))
    #
    # # print(newmodel)
    # pretrained_model = newmodel
    # csig = RegularizedClassSpecificImageGeneration(pretrained_model, top_filters_indices)
    # csig.generate()


    # model_path = os.path.join(DIRECTORYMODEL, f'model{6}.pth')
    #
    # model.load_state_dict(torch.load(model_path))
    # model.eval()
    # model.to('cpu')
    #
    # import torch
    # import torch.nn as nn
    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    # # Select the last convolutional layer and the corresponding SiLU activation layer
    # conv_layer = model.features[8][0]
    # silu_layer = model.features[8][2]
    #
    # # Get the weights of the selected convolutional layer
    # conv_weights = conv_layer.weight.data
    #
    # # Sum up the weights along each filter
    # filter_weights_sum = conv_weights.view(conv_weights.size(0), -1).sum(dim=1)
    #
    # # Get indices of filters with highest weights
    # num_top_filters = 5
    #
    # top_filters_indices = torch.argsort(filter_weights_sum, descending=True)[:num_top_filters]
    #
    #
    # # Inversion algorithm
    # def invert_image(model, layer, filter_indices, num_iterations=100, lr=0.1):
    #     synthesized_image = torch.randn(1, 3, 255, 255, requires_grad=True)
    #
    #     optimizer = torch.optim.Adam([synthesized_image], lr=lr)
    #
    #     for i in range(num_iterations):
    #         optimizer.zero_grad()
    #
    #         # Forward pass through the model to the specified layer
    #         x = synthesized_image
    #         for idx, module in enumerate(model.features[0:9]):
    #             x = module(x)
    #             if idx == layer:
    #                 break
    #
    #         # Get the output of the SiLU activation
    #         activation_output = x[0, filter_indices]
    #
    #         # Maximize the mean activation of the selected filters
    #         loss = -torch.mean(activation_output)
    #         loss.backward()
    #
    #         optimizer.step()
    #
    #     return synthesized_image.detach().numpy()
    #
    # # Generate synthetic images for the selected filters
    # synthetic_images = []
    # synthesized_image = invert_image(model, 8, top_filters_indices)
    # # for idx in top_filters_indices:
    # #     synthetic_images.append(synthesized_image)
    #
    # # Display the synthetic images
    # plt.imshow(np.transpose(synthesized_image[0], (1, 2, 0)))
    # plt.axis('off')
    # plt.title('Combined Top 5 Filters')
    # plt.show()
