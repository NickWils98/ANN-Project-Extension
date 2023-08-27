import torch

from RegularizedUnitSpecificImageGeneration import RegularizedClassSpecificImageGeneration


# function to calculate the model inversion
def model_inversion(model, model_path, iteration, subdir):
    # load the right weights in the model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Select the last convolutional layer
    conv_layer = model.features[8][0]
    # Get the weights of the selected convolutional layer
    conv_weights = conv_layer.weight.data

    # Sum up the weights along each filter
    filter_weights_sum = conv_weights.view(conv_weights.size(0), -1).sum(dim=1)
    # get the highest scoring filters
    indexed_list = list(enumerate(filter_weights_sum))
    sorted_indexed_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)
    top_filters_indices = [index for index, _ in sorted_indexed_list[:5]]
    # remove the last layers to expose the SiLu layer of the last block
    newmodel = torch.nn.Sequential(*(list(model.children())[:-2]))
    # use the RegularizedClassSpecificImageGeneration to create an image for each filter
    for i in top_filters_indices:
        print(f"filter: {i}")
        csig = RegularizedClassSpecificImageGeneration(newmodel, i, subdir)
        csig.generate(iterations=iteration)
