import torch
import numpy as np
from typing import Union, List, Tuple, Optional
import pytest

def optimize_filters_by_cropping(filters, threshold=1e-6):
    """
    Crop filters to remove near-zero values at edges for efficiency.
    
    Args:
        filters: Complex filters tensor of shape (n_freq_bins, win_size)
        threshold: Value below which filter coefficients are considered negligible
        
    Returns:
        cropped_filters: List of cropped filters
        crop_indices: List of (start, end) indices for each filter
    """
    cropped_filters = []
    crop_indices = []
    # print('filters shape: ', filters.shape)
    for i, filt in enumerate(filters):
        # Find significant values above threshold
        significant = torch.abs(filt) > threshold
        
        if significant.any():
            # Convert boolean tensor to indices and find first/last True value
            significant_indices = torch.nonzero(significant, as_tuple=False).view(-1)
            # print(significant_indices.shape)
            if significant_indices.numel() == 0:
                cropped_filters.append(filt[0:3])
                crop_indices.append((0, 3))
                continue

            start = significant_indices[0].item()
            end = significant_indices[-1].item() + 1  # Add 1 to include the last element
            
            # Ensure minimum size
            if end - start < 3:
                padding = 3 - (end - start)
                start = max(0, start - padding//2)
                end = min(len(significant), end + padding//2 + padding%2)
                
            cropped_filters.append(filt[start:end])
            crop_indices.append((start, end))
        else:
            # Handle all-zero filters
            cropped_filters.append(filt[0:3])  # Keep minimal size
            crop_indices.append((0, 3))
            
    return cropped_filters, crop_indices

def convolve_with_cropped_filters(
    arr, 
    cropped_kernels, 
    crop_indices, 
    padding='same'
):
    """
    Convolve signal with cropped kernels for efficiency
    
    Args:
        arr: Signal tensor of shape (n_channels, n_samples)
        cropped_kernels: List of cropped filter tensors
        crop_indices: List of (start, end) indices for each filter
        take_abs: Whether to return magnitude
        padding: Padding mode ('same' or 'valid')
        
    Returns:
        Output of shape (n_channels, n_samples, n_kernels)
    """

    if arr.ndim == 3:
        arr = arr.squeeze(1)
    # print('arr shape: ', arr.shape)
    n_channels, n_samples = arr.shape
    n_kernels = len(cropped_kernels)
    
    # Get output size based on padding mode
    if padding == 'same':
        out_size = n_samples
    elif padding == 'valid':
        # Find the largest kernel size for 'valid' mode
        max_kernel_size = max(end - start for start, end in crop_indices)
        out_size = n_samples - max_kernel_size + 1
    else:
        raise ValueError(f"Unsupported padding mode: {padding}")
    # print('out size: ', out_size)
    
    # Initialize output tensor with appropriate type
    is_complex = isinstance(cropped_kernels[0], torch.Tensor) and cropped_kernels[0].is_complex()
    out_dtype = torch.complex64 if is_complex else torch.float32
    
    n_channels = int(n_channels)
    n_kernels = int(n_kernels)

    if isinstance(out_size, torch.Size):
        out_size = tuple(out_size)

    if isinstance(out_size, int):
        output_shape = (n_channels, out_size, n_kernels)
    else:
        output_shape = (n_channels, *map(int, out_size), n_kernels)
        
    output = torch.zeros(
                        output_shape,
                        dtype=out_dtype,
                        device=arr.device)
    # print('output shape: ', output.shape)

    for k, (kernel, (start, end)) in enumerate(zip(cropped_kernels, crop_indices)):
        # print('k_index: ', k)
        kernel_size = end - start
        # print('kernel_size: ', kernel_size)

        # Flip kernel for convolution (since torch.nn.functional.conv1d does correlation)
        if isinstance(kernel, torch.Tensor):
            flipped_kernel = torch.flip(kernel, dims=[-1])
        else:
            # Handle non-tensor kernels (like those from list)
            kernel = torch.tensor(kernel, device=arr.device)
            flipped_kernel = torch.flip(kernel, dims=[-1])
        # print('filpped kernel', flipped_kernel.shape)

        # Ensure kernel is properly shaped for conv1d: (out_channels, in_channels, kernel_size)
        if flipped_kernel.dim() == 1:
            flipped_kernel = flipped_kernel.reshape(1, 1, -1)
        elif flipped_kernel.dim() == 2:
            # Assuming first dimension is out_channels
            flipped_kernel = flipped_kernel.unsqueeze(1)
        
        # Calculate padding based on mode
        if padding == 'same':

            # For 'same', we need to pad such that output size equals input size
            total_padding = kernel_size - 1
            pad_left = total_padding // 2
            pad_right = total_padding - pad_left

            # pad_size = (kernel_size - 1) // 2
            
            # # Handle even kernel sizes (need asymmetric padding)
            # pad_left = pad_size
            # pad_right = pad_size if kernel_size % 2 == 1 else pad_size - 1
            
            # Apply padding to input
            if pad_left > 0 or pad_right > 0:
                padded_arr = torch.nn.functional.pad(arr, (pad_left, pad_right))
            else:
                padded_arr = arr
            # print('padded_arr: ', padded_arr.shape)

            # Perform convolution with no additional padding
            result = torch.nn.functional.conv1d(
                input=padded_arr[:, None, :],
                weight=flipped_kernel,
                padding=0
            )
        else:  # 'valid' mode
            # print('arr: ', arr.shape)
            result = torch.nn.functional.conv1d(
                input=arr[:, None, :].to(flipped_kernel.dtype),
                weight=flipped_kernel,
                padding=0
            )
        # print('result: ', result.shape)

        # Store result in output tensor
        # For 'valid' mode with different kernel sizes, we need to align them
        if padding == 'valid':
            # Largest kernel gives the smallest output
            result_offset = max(0, (max_kernel_size - kernel_size) // 2)
            result_length = min(result.shape[2], out_size - result_offset)
            output[:, result_offset:result_offset+result_length, k] = result[:, 0, :result_length]
        else:
            # print('result before squeeze: ', result.shape)
            # For 'same' mode, output should be aligned
            # output[:, :, k] = result
            # result.shape[2] pode ser maior do que out_size devido a arredondamentos
            if result.shape[2] > out_size:
                crop = (result.shape[2] - out_size) // 2
                result = result[:, :, crop:crop + out_size]
            elif result.shape[2] < out_size:
                # Pad to match expected size (rare, mas por segurança)
                pad_total = out_size - result.shape[2]
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                result = torch.nn.functional.pad(result, (pad_left, pad_right))
            output[:, :, k] = result[:, 0, :]
        
    return output.transpose(1, 2)

def optim_conv1d(
        input, 
        weights, 
        padding,
    ):
    cropped_kernels, cropped_indices = optimize_filters_by_cropping(weights)
    out = convolve_with_cropped_filters(input, cropped_kernels, cropped_indices, padding)
    return out

################################################################################################################################

## TEST
def generate_test_filters():
    return torch.tensor([
        [0.0, 0.0, 1.0, 0.5, 0.0, 0.0],        # Trimmable edges
        [0.001, 0.0001, 0.00001, 0.000001, 0.0, 0.0],     # Below threshold
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],                   # All zero
        [0.0, 2.0, 0.0, 3.0, 0.0, 0.0]               # Middle active
    ], dtype=torch.float32)

## test cropped sizes
def test_optimize_filters_by_cropping():
    filters = generate_test_filters()
    cropped_filters, crop_indices = optimize_filters_by_cropping(filters, threshold=1e-3)

    # Expected cropping:
    # 1. From index 2 to 4
    # 2. All below threshold → forced to size 3
    # 3. All zero → default to first 3
    # 4. From index 1 to 4
    print('cropped_filter: ', cropped_filters)
    assert len(cropped_filters) == 4
    assert cropped_filters[0].tolist() == [1.0, 0.5, 0.0]
    assert len(cropped_filters[1]) == 3
    assert len(cropped_filters[2]) == 3
    assert cropped_filters[3].tolist() == [2.0, 0.0, 3.0]

    # Check crop indices
    print('crop_indices: ', crop_indices)
    assert crop_indices[0] == (2, 5)
    assert crop_indices[1] == (0, 3)
    assert crop_indices[2] == (0, 3)
    assert crop_indices[3] == (1, 4)

## test paddings validation
def test_convolve_with_cropped_filters():
    signal = torch.tensor([
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],    # Channel 0
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]     # Channel 1
    ], dtype=torch.float32)

    filters = generate_test_filters()
    cropped, crop_indices = optimize_filters_by_cropping(filters, threshold=1e-3)

    # Test SAME padding
    out_same = convolve_with_cropped_filters(
        arr=signal,
        cropped_kernels=cropped,
        crop_indices=crop_indices,
        padding='same'
    )

    assert out_same.shape == (2, 6, len(cropped))  # (channels, samples, kernels)
    assert not torch.isnan(out_same).any()

    # Test VALID padding
    out_valid = convolve_with_cropped_filters(
        arr=signal,
        cropped_kernels=cropped,
        crop_indices=crop_indices,
        padding='valid'
    )

    max_kernel = max(end - start for start, end in crop_indices)
    expected_len = signal.shape[1] - max_kernel + 1
    assert out_valid.shape == (2, expected_len, len(cropped))

## test complex values
def test_complex_filter_cropping_and_convolution():
    filters = torch.tensor([
        [0+0j, 1+1j, 0+0j, 2+2j, 0+0j]
    ], dtype=torch.complex64)

    cropped, indices = optimize_filters_by_cropping(filters, threshold=1e-6)
    assert len(cropped[0]) >= 3

    signal = torch.randn(1, 10, dtype=torch.complex64)
    out = convolve_with_cropped_filters(signal, cropped, indices, padding='same')
    assert out.shape == (1, 10, 1)
