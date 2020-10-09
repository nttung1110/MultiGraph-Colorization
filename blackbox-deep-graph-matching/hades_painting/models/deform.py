from __future__ import absolute_import, division

import torch
from torch.autograd import Variable

import numpy as np
from scipy.ndimage.interpolation import map_coordinates as sp_map_coordinates


def th_flatten(a):
    """Flatten tensor"""
    return a.contiguous().view(a.nelement())


def th_repeat(a, repeats, axis=0):
    """Torch version of np.repeat for 1D"""
    assert len(a.size()) == 1
    return th_flatten(torch.transpose(a.repeat(repeats, 1), 0, 1))


def np_repeat_2d(a, repeats):
    """TensorFlow version of np.repeat for 2D"""
    assert len(a.shape) == 2
    a = np.expand_dims(a, 0)
    a = np.tile(a, [repeats, 1, 1])
    return a


def th_gather_2d(inputs, coords):
    index = coords[:, 0] * inputs.size(1) + coords[:, 1]
    x = torch.index_select(th_flatten(inputs), 0, index)
    return x.view(coords.size(0))


def th_map_coordinates(inputs, coords, order=1):
    """Tensorflow verion of scipy.ndimage.map_coordinates
    Note that coords is transposed and only 2D is supported
    Parameters
    ----------
    inputs: tf.Tensor. shape = (s, s)
    coords: tf.Tensor. shape = (n_points, 2)
    order:
    """

    assert order == 1
    input_size = inputs.size(0)

    coords = torch.clamp(coords, 0, input_size - 1)
    coords_lt = coords.floor().long()
    coords_rb = coords.ceil().long()
    coords_lb = torch.stack([coords_lt[:, 0], coords_rb[:, 1]], 1)
    coords_rt = torch.stack([coords_rb[:, 0], coords_lt[:, 1]], 1)

    vals_lt = th_gather_2d(inputs, coords_lt.detach())
    vals_rb = th_gather_2d(inputs, coords_rb.detach())
    vals_lb = th_gather_2d(inputs, coords_lb.detach())
    vals_rt = th_gather_2d(inputs, coords_rt.detach())

    coords_offset_lt = coords - coords_lt.type(coords.data.type())

    vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[:, 0]
    vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[:, 0]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[:, 1]
    return mapped_vals


def sp_batch_map_coordinates(inputs, coords):
    """Reference implementation for batch_map_coordinates"""
    # coords = coords.clip(0, inputs.shape[1] - 1)

    assert (coords.shape[2] == 2)
    height = coords[:, :, 0].clip(0, inputs.shape[1] - 1)
    width = coords[:, :, 1].clip(0, inputs.shape[2] - 1)
    np.concatenate((np.expand_dims(height, axis=2), np.expand_dims(width, axis=2)), 2)

    mapped_vals = np.array([
        sp_map_coordinates(one_input, coord.T, mode='nearest', order=1)
        for one_input, coord in zip(inputs, coords)
    ])
    return mapped_vals


def th_batch_map_coordinates(inputs, coords, order=1):
    """Batch version of th_map_coordinates
    Only supports 2D feature maps
    Parameters
    ----------
    inputs: tf.Tensor. shape = (b, s, s)
    coords: tf.Tensor. shape = (b, n_points, 2)
    order:
    Returns
    -------
    tf.Tensor. shape = (b, s, s)
    """

    batch_size = inputs.size(0)
    input_height = inputs.size(1)
    input_width = inputs.size(2)

    n_coords = coords.size(1)

    # coords = torch.clamp(coords, 0, input_size - 1)

    coords = torch.cat((torch.clamp(coords.narrow(2, 0, 1), 0, input_height - 1), torch.clamp(coords.narrow(2, 1, 1), 0, input_width - 1)), 2)

    assert (coords.size(1) == n_coords)

    coords_lt = coords.floor().long()
    coords_rb = coords.ceil().long()
    coords_lb = torch.stack([coords_lt[..., 0], coords_rb[..., 1]], 2)
    coords_rt = torch.stack([coords_rb[..., 0], coords_lt[..., 1]], 2)
    idx = th_repeat(torch.arange(0, batch_size), n_coords).long()
    idx = Variable(idx, requires_grad=False)
    if inputs.is_cuda:
        idx = idx.cuda()

    def _get_vals_by_coords(tensor, tensor_coords):
        indices = torch.stack([
            idx, th_flatten(tensor_coords[..., 0]), th_flatten(tensor_coords[..., 1])
        ], 1)
        index = indices[:, 0] * tensor.size(1) * tensor.size(2) + indices[:, 1] * tensor.size(2) + indices[:, 2]
        vals = th_flatten(tensor).index_select(0, index)
        vals = vals.view(batch_size, n_coords)
        return vals

    vals_lt = _get_vals_by_coords(inputs, coords_lt.detach())
    vals_rb = _get_vals_by_coords(inputs, coords_rb.detach())
    vals_lb = _get_vals_by_coords(inputs, coords_lb.detach())
    vals_rt = _get_vals_by_coords(inputs, coords_rt.detach())

    coords_offset_lt = coords - coords_lt.type(coords.data.type())
    vals_t = coords_offset_lt[..., 0]*(vals_rt - vals_lt) + vals_lt
    vals_b = coords_offset_lt[..., 0]*(vals_rb - vals_lb) + vals_lb
    mapped_vals = coords_offset_lt[..., 1]* (vals_b - vals_t) + vals_t
    return mapped_vals


def sp_batch_map_offsets(inputs, offsets):
    """Reference implementation for tf_batch_map_offsets"""
    batch_size = inputs.shape[0]
    input_height = inputs.shape[1]
    input_width = inputs.shape[2]

    offsets = offsets.reshape(batch_size, -1, 2)
    grid = np.stack(np.mgrid[:input_height, :input_width], -1).reshape(-1, 2)
    grid = np.repeat([grid], batch_size, axis=0)
    coords = offsets + grid
    # coords = coords.clip(0, input_size - 1)

    mapped_vals = sp_batch_map_coordinates(inputs, coords)
    return mapped_vals


def th_generate_grid(batch_size, input_height, input_width, data_type, cuda):
    grid = np.meshgrid(
        range(input_height), range(input_width), indexing='ij'
    )
    grid = np.stack(grid, axis=-1)
    grid = grid.reshape(-1, 2)

    grid = np_repeat_2d(grid, batch_size)
    grid = torch.from_numpy(grid).type(data_type)
    if cuda:
        grid = grid.cuda()
    return Variable(grid, requires_grad=False)


def th_batch_map_offsets(inputs, offsets, grid=None, order=1):
    """Batch map offsets into input
    Parameters
    ---------
    inputs: torch.Tensor. shape = (b, s, s)
    grid:
    offsets: torch.Tensor. shape = (b, s, s, 2)
    Returns
    -------
    torch.Tensor. shape = (b, s, s)
    """
    batch_size = inputs.size(0)
    input_height = inputs.size(1)
    input_width = inputs.size(2)

    offsets = offsets.view(batch_size, -1, 2)
    if grid is None:
        grid = th_generate_grid(batch_size, input_height, input_width, offsets.data.type(), offsets.data.is_cuda)

    coords = offsets + grid

    mapped_vals = th_batch_map_coordinates(inputs, coords)
    return mapped_vals
