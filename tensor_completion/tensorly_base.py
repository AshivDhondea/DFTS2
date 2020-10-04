"""
Created on Tue Sep  1 09:37:27 2020.

Functions copied from Tensorly base.
Included to manipulate tensors in DFTS experiments run on ComputeCanada
clusters.

Adapted from
https://github.com/tensorly/tensorly/blob/master/tensorly/base.py


@article{tensorly,
  author  = {Jean Kossaifi and Yannis Panagakis and Anima Anandkumar and Maja Pantic},
  title   = {TensorLy: Tensor Learning in Python},
  journal = {Journal of Machine Learning Research},
  year    = {2019},
  volume  = {20},
  number  = {26},
  pages   = {1-6},
  url     = {http://jmlr.org/papers/v20/18-277.html}
}

"""
import numpy as np

# --------------------------------------------------------------------------- #
def fold(unfolded_tensor, mode, shape):
    """
        Refold the mode-`mode` unfolding into a tensor of shape `shape`.

        In other words, refolds the n-mode unfolded tensor
        into the original tensor of the specified shape.

    Parameters
    ----------
    unfolded_tensor : ndarray
        unfolded tensor of shape ``(shape[mode], -1)``
    mode : int
        the mode of the unfolding
    shape : tuple
        shape of the original tensor before unfolding

    Returns
    -------
    ndarray
        folded_tensor of shape `shape`
    """
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    return np.moveaxis(np.reshape(unfolded_tensor, full_shape), 0, mode)


def unfold(tensor, mode):
    """
    Return the mode-`mode` unfolding of `tensor` with modes starting at `0`.
    
    Parameters
    ----------
    tensor : ndarray
    mode : int, default is 0
           indexing starts at 0, therefore mode is in ``range(0, tensor.ndim)``
    
    Returns
    -------
    ndarray
        unfolded_tensor of shape ``(tensor.shape[mode], -1)``
    """
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))
# --------------------------------------------------------------------------- #