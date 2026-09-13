import torch
_CACHED_DEVICE = None

def get_cached_device() -> torch.device:
    """
    第一次调用的时候初始化，后面就一直返回cache的device
    """
    global _CACHED_DEVICE
    if _CACHED_DEVICE is not None:
        return _CACHED_DEVICE

    if torch.cuda.is_available():
        _CACHED_DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        _CACHED_DEVICE = torch.device("mps")
    else:
        _CACHED_DEVICE = torch.device("cpu")

    print("===== _CACHED_DEVICE: ======", _CACHED_DEVICE)
    return _CACHED_DEVICE
