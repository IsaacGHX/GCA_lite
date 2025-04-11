import torch

def setup_device(device):
    if isinstance(device, list) and len(device) == 1:
        device = torch.device(f'cuda:{device[0]}')
    else:
        device = None

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA:", torch.cuda.get_device_name(0))
        elif torch.backends.mps.is_available():
            device = torch.device("mps")  # For Apple Silicon
            print("Using MPS (Apple Silicon GPU)")
        else:
            device = torch.device("cpu")
            print("Using CPU")

    return device
