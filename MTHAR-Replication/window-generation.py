import numpy as np

def generate_windows(feature_length, base_length, scales):
    """
    Generate temporal anchor windows centered on each feature position.

    Args:
        feature_length (int): Length of backbone feature sequence (T)
        base_length (float): Base window length (n)
        scales (list): List of scale factors (e.g., [1, 2, 4])

    Returns:
        windows (np.ndarray): shape (num_windows, 2)
                              each row = [center, length]
    """
    windows = []

    for center in range(feature_length):
        for s in scales:
            l1 = base_length * np.sqrt(s)
            l2 = base_length / np.sqrt(s)

            windows.append([center, l1])
            windows.append([center, l2])

    return np.array(windows)