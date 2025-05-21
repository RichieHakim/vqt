from typing import Union, List, Tuple, Optional, Dict, Any, Sequence, Iterable, Type, Callable
import math

import torch
# import scipy.signal
from tqdm import tqdm
# import scipy.fftpack

from . import helpers
from . import optim

class VQT(torch.nn.Module):
    """
    Variable Q Transform. Class for applying the variable Q transform to
    signals. \n

    This function works differently than the VQT from librosa or nnAudio.
    This one does not use iterative lowpass filtering. \n 
    If fft_conv is False, then it uses a fixed set of filters, a Hilbert
    transform to compute the analytic signal, and then takes the magnitude. \n 
    If fft_conv is True, then it uses FFT convolution to compute the transform.
    \n
    
    Uses Pytorch for GPU acceleration, and allows gradients to pass through. \n

    Q: quality factor; roughly corresponds to the number of cycles in a
    filter. Here, Q is similar to the number of cycles within 4 sigma (95%)
    of a gaussian window. \n

    For running batches on GPU, transferring back to CPU tends to be the slowest
    part. \n

    RH 2022-2024

    Args:
        Fs_sample (float):
            Sampling frequency of the signal.
        Q_lowF (float):
            Q factor to use for the lowest frequency.
        Q_highF (float):
            Q factor to use for the highest frequency.
        F_min (float):
            Lowest frequency to use.
        F_max (float):
            Highest frequency to use.
        n_freq_bins (int):
            Number of frequency bins to use.
        win_size (int, None):
            Size of the window to use, in samples. \n
            If None, will be set to the next odd number after Q_lowF * (Fs_sample / F_min).
        window_type (str, np.ndarray, list, tuple):
            Window to use for the mother wavelet. \n
                * If string: Will be passed to scipy.signal.windows.get_window.
                See that documentation for options. Except for 'gaussian',
                which you should just pass the string 'gaussian' without any
                other arguments.
                * If array-like: Will be used as the window directly.
        symmetry (str):
            Whether to use a symmetric window or a single-sided window. \n
                * 'center': Use a symmetric / centered / 'two-sided' window.
                    \n
                * 'left': Use a one-sided, left-half window. Only left half
                of the filter will be nonzero. \n * 'right': Use a
                one-sided, right-half window. Only right half of the filter
                will be nonzero. \n
        taper_asymmetric (bool):
            Only used if symmetry is not 'center'. Whether to taper the
            center of the window by multiplying center sample of window by
            0.5.
        downsample_factor (int):
            Factor to downsample the signal by. If the length of the input
            signal is not divisible by downsample_factor, the signal will be
            zero-padded at the end so that it is.
        padding (str):
            Padding mode to use: \n
                * If fft_conv==False: ['valid', 'same'] \n
                * If fft_conv==True: ['full', 'valid', 'same'] \n
        conv_mode (str):
            Choose the convolutional processing mode: \n
                * ['fft_conv', 'normal_conv', 'optim_conv']
        fast_length (bool):
            Whether to use scipy.fftpack.next_fast_len to 
                find the next fast length for the FFT.
            This may be faster, but uses more memory.
        take_abs (bool):
            Whether to return the complex version of the transform. If
            True, then returns the absolute value (envelope) of the
            transform. If False, returns the complex transform.
        filters (Torch tensor):
            Filters to use. If None, will make new filters. Should be
            complex sinusoids. shape: (n_freq_bins, win_size)
        verbose (int):
            Verbosity. True to print warnings.
        plot_pref (bool):
            Whether to plot the filters.
    """
    def __init__(
        self,
        Fs_sample: Union[int, float]=1000,
        Q_lowF: Union[int, float]=1,
        Q_highF: Union[int, float]=20,
        F_min: Union[int, float]=1,
        F_max: Union[int, float]=400,
        n_freq_bins: int=50,
        win_size: Optional[int]=None,
        window_type: Union[str, torch.Tensor]='gaussian',
        symmetry: str='center',
        taper_asymmetric: bool=True,
        downsample_factor: int=4,
        padding: str='same',
        conv_mode: str='normal_conv',
        fast_length: bool=True,
        take_abs: bool=True,
        filters: Optional[torch.Tensor]=None,
        verbose: Union[int, bool]=True,
        plot_pref: bool=False,
    ):
        super().__init__()
        ## Prepare filters
        self.using_custom_filters = True if filters is not None else False
        self.filters = filters ## This line here is just for torch.jit.script to work. Delete it if you want to forget about jit.
        if filters is not None:
            ## Use provided filters
            self.filters = filters
        else:
            ## Make new filters
            filters_obj = helpers.VQT_filters(
                Fs_sample=Fs_sample,
                Q_lowF=Q_lowF,
                Q_highF=Q_highF,
                F_min=F_min,
                F_max=F_max,
                n_freq_bins=n_freq_bins,
                win_size=win_size,
                window_type=window_type,
                symmetry=symmetry,
                taper_asymmetric=taper_asymmetric,
                plot_pref=plot_pref,
            )
            self.filters, self.freqs, self.wins = filters_obj.make_filters()
        
        ## Get win_size from filters
        win_size = self.filters.shape[-1]

        ## Make filters the parameters of the model
        self.filters = torch.nn.Parameter(self.filters, requires_grad=False)

        ## Gather parameters from arguments
        (
            self.Fs_sample, 
            self.Q_lowF, 
            self.Q_highF, 
            self.F_min, 
            self.F_max, 
            self.n_freq_bins, 
            self.win_size, 
            self.downsample_factor, 
            self.padding, 
            self.conv_mode,
            self.fast_length,
            self.take_abs, 
            self.plot_pref, 
         ) = (
            Fs_sample, 
            Q_lowF, 
            Q_highF, 
            F_min, 
            F_max, 
            n_freq_bins, 
            win_size, 
            downsample_factor, 
            padding, 
            conv_mode,
            fast_length,
            take_abs, 
            plot_pref, 
         )
        
        ## Warnings
        if verbose >= 1:
            ## Warn if win_size is even
            if win_size % 2 != 1:
                print("Warning: win_size is even. This will result in a non-centered window. The x_axis will be offset by 0.5. It is recommended to use an odd win_size.")
            ## Warn if win_size is > 1024 to use fft_conv
            if win_size > 1024 and conv_mode != 'fft_conv':
                print(f"Warning: win_size is {win_size}, which is large for conv1d. Consider using conv_mode='fft_conv' for faster computation.")
            
    def forward(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of VQT.

        Args:
            X (Torch tensor):
                Input signal.
                shape: (n_channels, n_samples)

        Returns:
            Spectrogram (Torch tensor):
                Spectrogram of the input signal.
                shape: (n_channels, n_samples_ds, n_freq_bins)
        """
        assert isinstance(X, torch.Tensor), "X should be a torch tensor"
        X = X.type(torch.float32)

        ## Check that X and filters are on the same device
        assert X.device == self.filters.device, "X and filters should be on the same device"

        if X.ndim==1:
            X = X[None,:]

        assert X.ndim==2, "X should be 2D"  ## (n_channels, n_samples)
        assert self.filters.ndim==2, "Filters should be 2D" ## (n_freq_bins, win_size)
        # print('X: ', X.shape)
        ## Make spectrograms
        specs = downsample(
            X=convolve(
                arr=X, 
                kernels=self.filters, 
                take_abs=self.take_abs,
                conv_mode=self.conv_mode,
                padding=self.padding,
                fast_length=self.fast_length,
            ), 
            ds_factor=self.downsample_factor,
        )

        return specs
        
    def get_freqs(self) -> torch.Tensor:
        """
        Get the frequencies of the spectrogram.
        
        Args:
            None

        Returns:
            torch.Tensor:
                Frequencies of the spectrogram. \n
                shape: (n_freq_bins,)
        """
        assert hasattr(self, 'freqs'), "freqs not found. This should not happen."
        return self.freqs
    
    def get_xAxis(self, n_samples: int) -> torch.Tensor:
        """
        Get the x-axis for the spectrogram. \n
        RH 2024

        Args:
            n_samples (int):
                Number of samples in the signal.

        Returns:
            torch.Tensor:
                x-axis for the spectrogram in units of samples. \n
                shape: (n_samples_ds,)
        """
        ## Make x_axis
        x_axis = make_conv_xAxis(
            n_s=n_samples,
            n_k=self.filters.shape[-1],
            padding=self.padding,
            downsample_factor=self.downsample_factor,
            device='cpu',
        )
        return x_axis

    def __repr__(self):
        if self.using_custom_filters:
            return f"Optimized VQT with custom filters"
        else:
            attributes_to_print = []
            for k, v in self.__dict__.items():
                if (k not in ['filters', 'freqs', 'wins']) and (not k.startswith('_')) and (not callable(v)):
                    attributes_to_print.append(k)
            return f"VQT object with parameters: {''.join([f'{k}={getattr(self, k)}, ' for k in attributes_to_print])[:-2]}"
        

def downsample(
    X: torch.Tensor, 
    ds_factor: int=4, 
) -> torch.Tensor:
    """
    Downsample a signal using average pooling. \n
    If X is complex, it will be split into magnitude and phase, downsampled,
    and then recombined. \n

    RH 2024

    Args:
        X (torch.Tensor):
            Signal to downsample. \n
            ``shape``: (..., n_samples)
        ds_factor (int):
            Factor to downsample the signal by.

    Returns:
        torch.Tensor:
            Downsampled signal.
    """
    if ds_factor == 1:
        return X
    
    ## Assert X is a torch tensor
    assert isinstance(X, torch.Tensor), "X should be a torch tensor"
    ## Ensure X.ndim in [2, 3]
    if X.ndim==1:
        X = X[None,:]
    assert X.ndim in [2, 3], "X should be 2D or 3D"  ## (n_channels, n_samples)
    
    ## Check is X is complex
    if X.is_complex() == False:
        return _helper_ds(X, ds_factor=ds_factor)
    elif X.is_complex() == True:
        ## Unfortunately, torch.nn.functional.avg_pool1d does not support complex numbers. So we have to split it up into
        ##  phases and magnitudes (convert imaginary to polar, split, downsample, recombine with polar to complex conversion)
        X = _helper_imag_to_polarReal(X)
        X = torch.stack([_helper_ds(X[ii], ds_factor=ds_factor) for ii in range(2)], dim=0)
        X = _helper_polarReal_to_imag(X)
        return X

    else:
        raise ValueError("X should be a torch tensor of type float or complex")
def _helper_polarReal_to_imag(arr: torch.Tensor) -> torch.Tensor:
    return arr[0] * torch.exp(1j * arr[1])
def _helper_imag_to_polarReal(arr: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.abs(arr), torch.angle(arr)], dim=0)
def _helper_ds(arr: torch.Tensor, ds_factor: int) -> torch.Tensor:
    return torch.nn.functional.avg_pool1d(
        arr,
        kernel_size=[int(ds_factor)],
        stride=ds_factor,
        ceil_mode=True,
        # padding=0,
        count_include_pad=False,  ## Prevents edge effects
    )

def convolve(
    arr: torch.Tensor, 
    kernels: torch.Tensor, 
    take_abs: bool=False,
    padding: str='same', 
    conv_mode: str='normal_conv', 
    fast_length: bool=False,
) -> torch.Tensor:
    """
    Convolve a signal with a set of kernels. \n

    RH 2024

    Args:
        arr (torch.Tensor):
            Signal to convolve. \n
            ``shape``: (n_channels, n_samples)
        kernels (torch.Tensor):
            Kernels to convolve with. \n
            ``shape``: (n_kernels, win_size)
        take_abs (bool):
            Whether to take the absolute value of the result.
        padding (str):
            Padding mode to use: \n
                * If fft_conv==False: ['valid', 'same'] \n
                * If fft_conv==True: ['full', 'valid', 'same'] \n
        fft_conv (bool):
            Whether to use FFT convolution.
        fast_length (bool):
            Whether to use scipy.fftpack.next_fast_len to find the next fast
            length for the FFT.

    Returns:
        torch.Tensor:
            Result of the convolution. \n
            ``shape``: (n_channels, n_samples, n_kernels)
    """
    assert all(isinstance(arg, torch.Tensor) for arg in [arr, kernels]), "arr and kernels should be torch tensors"

    arr = arr[None,:] if arr.ndim==1 else arr
    kernels = kernels[None,:] if kernels.ndim==1 else kernels

    arr = arr[:,None,:]  ## Shape: (n_channels, 1, n_samples)
    # kernels = kernels  ## Shape: (n_kernels, win_size)
    # print('data in: ', arr.shape)
    
    if conv_mode == 'fft_conv':
        # print('fft_conv in')
        out = fftconvolve(
            x=arr,  
            y=(kernels)[None,:,:], 
            mode=padding,
            fast_length=fast_length,
        )
        # print('fft_conv out: ', out.shape)
    # elif conv_mode == 'optim_conv':
    #     print('optim_conv in')
    #     cropped_kernels, crop_indices = optim.optimize_filters_by_cropping(kernels)
    #     print('kernels list: ', len(cropped_kernels))
    #     out = optim.convolve_with_cropped_filters(
    #         arr=arr,
    #         cropped_kernels=cropped_kernels,
    #         crop_indices=crop_indices,
    #         padding=padding
    #     )
    #     print('optim_conv out: ', out.shape)
    else:
        flag_kernels_complex = kernels.is_complex()
        kernels = torch.flip(kernels, dims=[-1,])[:,None,:]  ## Flip because torch's conv1d uses cross-correlation, not convolution.
        
        if flag_kernels_complex:
            kernels_list = [torch.real(kernels), torch.imag(kernels)]
        else:
            kernels_list = [kernels,]
        # print('kernels list: ', [k.shape for k in kernels_list])

        if conv_mode == 'optim_conv':
            # print('optim_conv in')
            out_conv = [optim.optim_conv1d(
                input=arr, 
                weights=k.squeeze(1), 
                padding='same',
            ) for k in kernels_list]
            # print('optim_conv out: ', torch.stack(out_conv, dim=0).shape)
        else:
            # print('normal_conv in')
            out_conv = [torch.nn.functional.conv1d(
                input=arr, 
                weight=k, 
                padding=padding,
            ) for k in kernels_list]
            # print('normal_conv out: ', torch.stack(out_conv, dim=0).shape)

        if flag_kernels_complex:
            out = torch.complex(out_conv[0], out_conv[1])
        else:
            out = out_conv[0]
        
    if take_abs:
        out = torch.abs(out)

    # print('data out: ', out.shape)
    return out


def fftconvolve(
    x: torch.Tensor, 
    y: torch.Tensor, 
    mode: str='valid',
    fast_length: bool=False,
) -> torch.Tensor:
    """
    Convolution using the FFT method. \n
    This is adapted from of torchaudio.functional.fftconvolve that handles
    complex numbers. Code is added for handling complex inputs. \n
    NOTE: For mode='same' and y length even, torch's conv1d convention is used,
    which pads 1 more at the end and 1 fewer at the beginning (which is
    different from numpy/scipy's convolve). See apply_padding_mode for more
    details. \n

    RH 2024

    Args:
        x (torch.Tensor):
            First input. (signal) \n
            Convolution performed along the last dimension.
        y (torch.Tensor):
            Second input. (kernel) \n
            Convolution performed along the last dimension.
        mode (str):
            Padding mode to use. ['full', 'valid', 'same']
        fast_length (bool):
            Whether to use scipy.fftpack.next_fast_len to 
             find the next fast length for the FFT.
            Set to False if you want to use backpropagation.

    Returns:
        torch.Tensor:
            Result of the convolution. \n
            Padding applied to last dimension. \n
            ``shape``: (..., n_samples)
    """
    ## Compute the convolution
    n_original = x.shape[-1] + y.shape[-1] - 1
    # n = scipy.fftpack.next_fast_len(n_original) if fast_length else n_original
    n = next_fast_len(n_original) if fast_length else n_original
    # n = n_original
    f = torch.fft.fft(x, n=n, dim=-1) * torch.fft.fft(y, n=n, dim=-1)
    fftconv_xy = torch.fft.ifft(f, n=n, dim=-1)
    return apply_padding_mode(
        conv_result=fftconv_xy,
        x_length=x.shape[-1],
        y_length=y.shape[-1],
        mode=mode,
    )

## For some reason jit is slower here
# @torch.jit.script
def next_fast_len(size: int) -> int:
    """
    Taken from PyTorch Forecasting:
    Returns the next largest number ``n >= size`` whose prime factors are all
    2, 3, or 5. These sizes are efficient for fast fourier transforms.
    Equivalent to :func:`scipy.fftpack.next_fast_len`.

    Implementation from pyro

    :param int size: A positive number.
    :returns: A possibly larger number.
    :rtype int:
    """
    assert isinstance(size, int) and size > 0
    next_size = size
    while True:
        remaining = next_size
        for n in (2, 3, 5):
            while remaining % n == 0:
                remaining = remaining // n
        if remaining == 1:
            return next_size
        next_size += 1


def apply_padding_mode(
    conv_result: torch.Tensor, 
    x_length: int, 
    y_length: int, 
    mode: str = "valid",
) -> torch.Tensor:
    """
    This is adapted from torchaudio.functional._apply_convolve_mode. \n
    NOTE: This function has a slight change relative to torchaudio's version.
    For mode='same', ceil rounding is used. This results in fftconv matching the
    result of conv1d. However, this then results in it not matching the result of
    scipy.signal.fftconvolve. This is a tradeoff. The difference is only a shift
    in 1 sample when y_length is even. This phenomenon is a result of how conv1d
    handles padding, and the fact that conv1d is actually cross-correlation, not
    convolution. \n

    RH 2024

    Args:
        conv_result (torch.Tensor):
            Result of the convolution.
            Padding applied to last dimension.
        x_length (int):
            Length of the first input.
        y_length (int):
            Length of the second input.
        mode (str):
            Padding mode to use.

    Returns:
        torch.Tensor:
            Result of the convolution with the specified padding mode. \n
            ``shape``: (..., n_samples)
    """
    n = x_length + y_length - 1
    valid_convolve_modes = ["full", "valid", "same"]
    if mode == "full":
        return conv_result
    elif mode == "valid":
        len_target = max(x_length, y_length) - min(x_length, y_length) + 1
        idx_start = (n - len_target) // 2
        return conv_result[..., idx_start : idx_start + len_target]
    elif mode == "same":
        # idx_start = (conv_result.size(-1) - x_length) // 2  ## This is the original line from torchaudio
        idx_start = math.ceil((n - x_length) / 2)  ## This line is different from torchaudio
        return conv_result[..., idx_start : idx_start + x_length]
    else:
        raise ValueError(f"Unrecognized mode value '{mode}'. Please specify one of {valid_convolve_modes}.")


def make_conv_xAxis(
    n_s: int,
    n_k: int,
    padding: str='same',
    downsample_factor: int=4,
    device: torch.device='cpu',
) -> torch.Tensor:
    """
    Make the x-axis for the result of a convolution.
    This is adapted from torchaudio.functional._make_conv_xAxis.

    RH 2024

    Args:
        n_s (int):
            Length of the signal.
        n_k (int):
            Length of the kernel.
        padding (str):
            Padding mode to use.
        downsample_factor (int):
            Factor to downsample the signal by.
        device (str):
            Device to use.

    Returns:
        torch.Tensor:
            x-axis for the result of a convolution. \n
            ``shape``: (n_samples_ds,)
    """

    ## If n_k is odd, then no offset, if even, then offset by 0.5
    ### PyTorch's conv1d and for 'same' pads more to the right, so on the first index of the output the kernel is centered at an offset of 0.5
    offset = 0.5 if n_k % 2 == 0 else 0.0

    x_axis_full = torch.arange(
        -(n_k-1)//2 + offset,
        n_s + (n_k-1)//2 + offset,
        dtype=torch.float32,
        device=device,
    )
    ### Then, apply padding mode to it
    x_axis_padModed = apply_padding_mode(
        conv_result=x_axis_full,
        x_length=n_s,
        y_length=n_k,
        mode=padding,
    ).squeeze()
    ### Then, let's downsample it
    x_axis = downsample(
        X=x_axis_padModed[None,None,:],
        ds_factor=downsample_factor,
    ).squeeze().to(device)

    return x_axis