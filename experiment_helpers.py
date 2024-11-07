import hashlib

import torch


def reinitialize_model(model, seed=42):
    # Save the current random state
    state_backup = torch.get_rng_state()

    # Set the seed for reproducibility
    torch.manual_seed(seed)

    for layer in model.modules():
        if hasattr(layer, 'reset_parameters'):
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
