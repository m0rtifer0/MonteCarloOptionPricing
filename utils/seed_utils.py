import os
import random
import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across Python, NumPy, PyTorch, and TensorFlow.

    Parameters:
    -----------
    seed : int
        The seed value to use for all RNGs (default is 42).
    """

    # Set Python seed
    random.seed(seed)

    # Set NumPy seed
    np.random.seed(seed)

    # Set environment hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Set PyTorch seed if available
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set TensorFlow seed if available
    if tf is not None:
        tf.random.set_seed(seed)

    print(f"[INFO] Global seed set to {seed}")
