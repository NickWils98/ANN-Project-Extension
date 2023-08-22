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

def score_cam(model_path, val_loader, device, image_list, num_classes, plot_path):
    image_list = sorted(image_list)
    plot_path = os.path.join(plot_path, "ScoreCAM")
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
                    plot_save_counter = os.path.join(plot_path, f'scoreCAM{counter}.png')
                    plt.savefig(plot_save_counter, dpi=300)
                    plt.show()
                # Once you're finished, clear the hooks on your model
                scorecam.remove_hooks()
                if len(image_list)==0:
                    return
            counter += 1
            labelcounter += 1