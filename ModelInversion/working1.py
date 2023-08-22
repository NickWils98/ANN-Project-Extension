model_path = os.path.join(DIRECTORYMODEL, f'model{6}.pth')

    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to('cpu')

    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt

    # Select the last convolutional layer and the corresponding SiLU activation layer
    conv_layer = model.features[8][0]
    silu_layer = model.features[8][2]

    # Get the weights of the selected convolutional layer
    conv_weights = conv_layer.weight.data

    # Sum up the weights along each filter
    filter_weights_sum = conv_weights.view(conv_weights.size(0), -1).sum(dim=1)

    # Get indices of filters with highest weights
    num_top_filters = 5

    top_filters_indices = torch.argsort(filter_weights_sum, descending=True)[:num_top_filters]


    # Inversion algorithm
    def invert_image(model, layer, filter_indices, num_iterations=100, lr=0.1):
        synthesized_image = torch.randn(1, 3, 255, 255, requires_grad=True)

        optimizer = torch.optim.Adam([synthesized_image], lr=lr)

        for i in range(num_iterations):
            optimizer.zero_grad()

            # Forward pass through the model to the specified layer
            x = synthesized_image
            for idx, module in enumerate(model.features[0:9]):
                x = module(x)
                if idx == layer:
                    break

            # Get the output of the SiLU activation
            activation_output = x[0, filter_indices]

            # Maximize the mean activation of the selected filters
            loss = -torch.mean(activation_output)
            loss.backward()

            optimizer.step()

        return synthesized_image.detach().numpy()

    # Generate synthetic images for the selected filters
    synthetic_images = []
    synthesized_image = invert_image(model, 8, top_filters_indices)
    # for idx in top_filters_indices:
    #     synthetic_images.append(synthesized_image)

    # Display the synthetic images
    plt.imshow(np.transpose(synthesized_image[0], (1, 2, 0)))
    plt.axis('off')
    plt.title('Combined Top 5 Filters')
    plt.show()
