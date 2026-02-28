# python/compute_frequency.py

def combine_freqs(f_shape: float, f_mat: float,
                  w_shape: float = 0.8,
                  w_mat: float = 0.2) -> float:
    """
    Combine shape- and material-based frequency estimates.

    For now it's just a placeholder. Later we'll tune w_shape, w_mat.
    """
    return w_shape * f_shape + w_mat * f_mat
