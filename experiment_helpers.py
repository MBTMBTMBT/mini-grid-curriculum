import hashlib
import torch
import torch.nn as nn
import torch.nn.init as init


def reinitialize_model(model, seed=42):
    # Save the current random state
    state_backup = torch.get_rng_state()
    torch.manual_seed(seed)  # Set the seed for reproducibility

    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            # Use Kaiming initialization for fully connected layers
            init.kaiming_normal_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                init.constant_(layer.bias, 0)

        elif isinstance(layer, nn.Conv2d):
            # Use Kaiming initialization for convolutional layers
            init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                init.constant_(layer.bias, 0)

        elif isinstance(layer, nn.LSTM) or isinstance(layer, nn.GRU):
            # Use Xavier initialization for recurrent layers (LSTM/GRU)
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    init.xavier_uniform_(param)
                elif 'bias' in name:
                    init.constant_(param, 0)

        elif isinstance(layer, nn.TransformerEncoderLayer) or isinstance(layer, nn.TransformerDecoderLayer):
            # Use Xavier initialization for Transformer layers
            init.xavier_uniform_(layer.self_attn.in_proj_weight)
            init.xavier_uniform_(layer.self_attn.out_proj.weight)
            init.xavier_uniform_(layer.linear1.weight)
            init.xavier_uniform_(layer.linear2.weight)
            if layer.self_attn.in_proj_bias is not None:
                init.constant_(layer.self_attn.in_proj_bias, 0)
            if layer.self_attn.out_proj.bias is not None:
                init.constant_(layer.self_attn.out_proj.bias, 0)
            if layer.linear1.bias is not None:
                init.constant_(layer.linear1.bias, 0)
            if layer.linear2.bias is not None:
                init.constant_(layer.linear2.bias, 0)

        elif isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d):
            # Use constant initialization for BatchNorm layers
            init.constant_(layer.weight, 1)
            init.constant_(layer.bias, 0)

        elif hasattr(layer, 'reset_parameters'):
            # Call reset_parameters on any other layer that has this method
            layer.reset_parameters()

    # Restore the original random state
    torch.set_rng_state(state_backup)


def hash_model(model):
    hasher = hashlib.sha256()
    with torch.no_grad():
        for param in model.parameters():
            # Use the parameter's data to update the hash.
            # We ensure the data is on CPU and in a consistent byte order
            hasher.update(param.data.cpu().numpy().tobytes())
    return hasher.hexdigest()
