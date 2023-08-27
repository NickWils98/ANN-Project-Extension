import os

import matplotlib.pyplot as plt
import torch
from torchcam.methods import ScoreCAM
from torchcam.utils import overlay_mask
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_pil_image


# Function to calculate the ScoreCAM in batches for pretext models
def score_cam_batch(model, model_path, val_loader, device, image_list, plot_path, batch_nr):
    # sort list of images wanted for scoreCAM
    image_list = sorted(image_list)
    # path to save scorecam
    plot_path = os.path.join(plot_path, "ScoreCAM")
    # Make the subdirectory if it doesn't exist yet
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # target layers for scoreCAM
    target_layers = [1, 2, 8]

    counter = 0
    # load the right weights in the model
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)

    # go over all images
    for batch_rotated_images, labels in val_loader:

        labelcounter = 0
        all_images = []
        rotated_labels = []
        # go over all images in batch
        for j, rotated_image_list in enumerate(batch_rotated_images):
            if j == batch_nr:
                all_images.extend(rotated_image_list)
                rotated_labels.extend([j] * len(rotated_image_list))
        for img in all_images:
            # only go over image if the images is in the list
            if counter in image_list:
                # remove that image out that list
                image_list.remove(counter)
                # go over all the layers you want to view
                for i, target_layer in enumerate(target_layers):
                    # create scoreCAM
                    scorecam = ScoreCAM(model, target_layer=model.features[target_layer])
                    image = img.unsqueeze(0).to(device)

                    with torch.no_grad():
                        # get run image through the model
                        outputs = model(image)
                        x, predicted_class = torch.max(outputs.data, 1)
                    if i == 0:
                        print(f'index: {counter}, true label: {labels[labelcounter][batch_nr]}, guess: {predicted_class.item()}')

                    # Forward the image through the model to hook into the convolutional features
                    # Generate heatmap using Score-CAM
                    heatmap = scorecam(predicted_class.item(), predicted_class)
                    # overlay the heatmap on the image
                    for name, cam in zip(scorecam.target_names, heatmap):
                        result = overlay_mask(to_pil_image(img), to_pil_image(cam.squeeze(0), mode='F'), alpha=0.5)
                        plt.subplot(1, len(target_layers), i + 1)
                        plt.imshow(result)
                        plt.axis('off')
                        plt.title(f'Layer {target_layer + 1}')
                    scorecam.remove_hooks()

                plt.tight_layout()
                # show and save the image
                plot_save_counter = os.path.join(plot_path, f'scoreCAM{counter}.png')
                plt.savefig(plot_save_counter, dpi=300)
                plt.show()
                # if all the images are shown stop
                if len(image_list) == 0:
                    return
            counter += 1
            labelcounter += 1



# Function to calculate the ScoreCAM
def score_cam(model, model_path, val_loader, orignal_data, device, image_list, plot_path):
    # sort list of images wanted for scoreCAM
    image_list = sorted(image_list)
    # path to save scorecam
    plot_path = os.path.join(plot_path, "ScoreCAM")
    # Make the subdirectory if it doesn't exist yet
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    counter = 0
    # load the right weights in the model
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)

    # tranformer for images for the dataset without changing colors
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    # load unaltered dataset with transformer
    val_dataset = ImageFolder(os.path.join(orignal_data, "validation"), transform=transform)
    # target layers for scoreCAM
    target_layers = [1, 2, 8]

    # go over all images
    for batch_images, labels in val_loader:
        labelcounter = 0
        # go over all images in batch
        for img in batch_images:
            # only go over image if the images is in the list
            if counter in image_list:
                # remove that image out that list
                image_list.remove(counter)
                # go over all the layers you want to view
                for i, target_layer in enumerate(target_layers):
                    # create scoreCAM
                    scorecam = ScoreCAM(model, target_layer=model.features[target_layer])
                    og_image = val_dataset[counter][0]
                    image = img.unsqueeze(0).to(device)

                    with torch.no_grad():
                        # get run image through the model
                        outputs = model(image)
                        x, predicted_class = torch.max(outputs.data, 1)
                    if i == 0:
                        print(f'index: {counter}, true label: {labels[labelcounter]}, guess: {predicted_class.item()}')

                    # Generate heatmap using Score-CAM
                    heatmap = scorecam(predicted_class.item(), predicted_class)
                    # overlay the heatmap on the image
                    for name, cam in zip(scorecam.target_names, heatmap):
                        result = overlay_mask(to_pil_image(og_image), to_pil_image(cam.squeeze(0), mode='F'), alpha=0.5)
                        plt.subplot(1, len(target_layers), i + 1)
                        plt.imshow(result)
                        plt.axis('off')
                        plt.title(f'Layer {target_layer}')
                    scorecam.remove_hooks()
                plt.tight_layout()
                # show and save the image
                plot_save_counter = os.path.join(plot_path, f'scoreCAM{counter}.png')
                plt.savefig(plot_save_counter, dpi=300)
                plt.show()

                # if all the images are shown stop
                if len(image_list) == 0:
                    return
            counter += 1
            labelcounter += 1
