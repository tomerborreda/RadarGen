import torch.nn as nn
from torch.nn import Conv2d
from torch.nn.parameter import Parameter


def inflate_sana_input_channels_for_radargen(model, logger, num_conditions: int = 3):
    weight_replication_factor = num_conditions + 1 # +1 for the noised latent
    _weight = model.x_embedder.proj.weight.clone()
    _bias = model.x_embedder.proj.bias.clone()
    _weight = _weight.repeat((1, weight_replication_factor, 1, 1))  # Keep selected channel(s)
    
    # Scale the weights to maintain activation magnitude
    _weight = _weight / float(weight_replication_factor)

    # New conv_in channel
    _weight_channels = _weight.shape[1] + model.modality_pe.shape[1] # Add C_pe for the modality PE
    _n_convin_out_channel = model.x_embedder.proj.out_channels
    _kernel_size = model.x_embedder.proj.kernel_size
    _stride = model.x_embedder.proj.stride
    _padding = model.x_embedder.proj.padding
    _new_conv_in = Conv2d(
        _weight_channels, _n_convin_out_channel, kernel_size=_kernel_size, stride=_stride, padding=_padding, bias=True
    )
    # Initialize the weight like in the original SANA model
    w = _new_conv_in.weight.data
    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    # Set the value of the weight and bias as the original, divided by the number of duplications
    _new_conv_in.weight.data[:, :_weight.shape[1]] = _weight
    _new_conv_in.bias = Parameter(_bias)
    _new_conv_in.to(model.x_embedder.proj.weight.device, model.x_embedder.proj.weight.dtype)

    # Finally, replace the original layer with the new one
    model.x_embedder.proj = _new_conv_in
    logger.info("SANA PatchEmbedMS.proj layer was replaced")
