import math

import torch
import numpy as np
import scipy.signal
from tqdm import tqdm
import scipy.fftpack

from . import helpers

class VQT():
    def __init__(
        self,
        Fs_sample=1000,
        Q_lowF=3,
        Q_highF=20,
        F_min=10,
        F_max=400,
        n_freq_bins=55,
        win_size=501,
        symmetry='center',
        taper_asymmetric=True,
        downsample_factor=4,
        padding='valid',
        fft_conv=False,
        fast_length=True,
        DEVICE_compute='cpu',
        DEVICE_return='cpu',
        batch_size=1000,
        return_complex=False,
        filters=None,
        verbose=True,
        plot_pref=False,
        progressBar=True,
    ):
        """
        Variable Q Transform. Class for applying the variable Q transform to
        signals.

        This function works differently than the VQT from librosa or nnAudio.
        This one does not use iterative lowpass filtering. \n If fft_conv is
        False, then it uses a fixed set of filters, a Hilbert transform to
        compute the analytic signal, and then takes the magnitude. \n If
        fft_conv is True, then it uses FFT convolution to compute the transform.
        \n
        
        Uses Pytorch for GPU acceleration, and allows gradients to pass through.
        \n

        Q: quality factor; roughly corresponds to the number of cycles in a
        filter. Here, Q is the number of cycles within 4 sigma (95%) of a
        gaussian window. \n

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
            win_size (int):
                Size of the window to use, in samples.
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
                signal is not
                 divisible by downsample_factor, the signal will be zero-padded
                 at the end so that it is.
            padding (str):
                Padding mode to use: \n
                    * If fft_conv==False: ['valid', 'same'] \n
                    * If fft_conv==True: ['full', 'valid', 'same'] \n
            fft_conv (bool):
                Whether to use FFT convolution. This is faster, but may be less
                accurate. If False, uses torch's conv1d.
            fast_length (bool):
                Whether to use scipy.fftpack.next_fast_len to 
                 find the next fast length for the FFT.
                This may be faster, but uses more memory.
            DEVICE_compute (str):
                Device to use for computation.
            DEVICE_return (str):
                Device to use for returning the results.
            batch_size (int):
                Number of signals to process at once. Use a smaller batch size
                if you run out of memory.
            return_complex (bool):
                Whether to return the complex version of the transform. If
                False, then returns the absolute value (envelope) of the
                transform. downsample_factor must be 1 if this is True.
            filters (Torch tensor):
                Filters to use. If None, will make new filters. Should be
                complex sinusoids. shape: (n_freq_bins, win_size)
            verbose (int):
                Verbosity. True to print warnings.
            plot_pref (bool):
                Whether to plot the filters.
            progressBar (bool):
                Whether to show a progress bar.
        """
        ## Prepare filters
        if filters is not None:
            ## Use provided filters
            self.using_custom_filters = True
            self.filters = filters
        else:
            ## Make new filters
            self.using_custom_filters = False
            self.filters, self.freqs, self.wins = helpers.make_VQT_filters(
                Fs_sample=Fs_sample,
                Q_lowF=Q_lowF,
                Q_highF=Q_highF,
                F_min=F_min,
                F_max=F_max,
                n_freq_bins=n_freq_bins,
                win_size=win_size,
                symmetry=symmetry,
                taper_asymmetric=taper_asymmetric,
                plot_pref=plot_pref,
            )

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
            self.fft_conv,
            self.fast_length,
            self.DEVICE_compute, 
            self.DEVICE_return, 
            self.batch_size, 
            self.return_complex, 
            self.plot_pref, 
            self.progressBar,
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
            fft_conv,
            fast_length,
            DEVICE_compute, 
            DEVICE_return, 
            batch_size, 
            return_complex, 
            plot_pref, 
            progressBar,
         )
        
        ## Warnings
        if verbose >= 1:
            ## Warn if win_size is even
            if win_size % 2 != 1:
                print("Warning: win_size is even. This will result in a non-centered window. The x_axis will be offset by 0.5. It is recommended to use an odd win_size.")
            ## Warn if win_size is > 1024 to use fft_conv
            if win_size > 1024 and fft_conv == False:
                print(f"Warning: win_size is {win_size}, which is large for conv1d. Consider using fft_conv=True for faster computation.")
            
    def __call__(self, X):
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
            x_axis (Torch tensor):
                New x-axis for the spectrogram in units of samples.
                Get units of time by dividing by Fs_sample.
            self.freqs (Torch tensor):
                Frequencies of the spectrogram.
        """
        if type(X) is not torch.Tensor:
            X = torch.as_tensor(X, dtype=torch.float32, device=self.DEVICE_compute)

        if X.ndim==1:
            X = X[None,:]

        assert X.ndim==2, "X should be 2D"  ## (n_channels, n_samples)
        assert self.filters.ndim==2, "Filters should be 2D" ## (n_freq_bins, win_size)

        ## Make iterator for batches
        batches = helpers.make_batches(X, batch_size=self.batch_size, length=X.shape[0])

        ## Make function to apply to signals
        fn_vqt = lambda arr, filters: downsample(
            X=convolve(
                arr=arr, 
                kernels=filters, 
                take_abs=(self.return_complex==False),
                fft_conv=self.fft_conv,
                padding=self.padding,
                fast_length=self.fast_length,
                DEVICE=self.DEVICE_compute
            ), 
            ds_factor=self.downsample_factor,
            return_complex=self.return_complex,
        ).to(self.DEVICE_return)

        ## Make spectrograms
        specs = torch.cat([fn_vqt(arr=arr, filters=self.filters) for arr in tqdm(batches, disable=(self.progressBar==False), leave=True, total=int(np.ceil(X.shape[0]/self.batch_size)))])

        ## Make x_axis
        x_axis = make_conv_xAxis(
            n_s=X.shape[-1],
            n_k=self.filters.shape[-1],
            padding=self.padding,
            downsample_factor=self.downsample_factor,
            DEVICE_compute=self.DEVICE_compute,
            DEVICE_return=self.DEVICE_return,
        )

        return specs, x_axis, self.freqs

    def __repr__(self):
        if self.using_custom_filters:
            return f"VQT with custom filters"
        else:
            return f"VQT object with parameters: {''.join([f'{k}={getattr(self, k)}, ' for k, v in self.__dict__.items() if k not in ['filters', 'freqs', 'wins']])[:-2]}"
        

def downsample(
    X: torch.Tensor, 
    ds_factor: int=4, 
    return_complex: bool=False,
):
    """
    Downsample a signal using average pooling. \n

    RH 2024

    Args:
        X (torch.Tensor):
            Signal to downsample. \n
            ``shape``: (..., n_samples)
        ds_factor (int):
            Factor to downsample the signal by.
        return_complex (bool):
            Whether to return the complex version of the signal.

    Returns:
        torch.Tensor:
            Downsampled signal.
    """
    if ds_factor == 1:
        return X
    
    fn_ds = lambda arr: torch.nn.functional.avg_pool1d(
        arr,
        kernel_size=[int(ds_factor)],
        stride=ds_factor,
        ceil_mode=True,
    )
    if return_complex == False:
        return fn_ds(X)
    elif return_complex == True:
        ## Unfortunately, torch.nn.functional.avg_pool1d does not support complex numbers. So we have to split it up.
        ### Split X, shape: (..., n_samples) into real and imaginary parts, shape: (..., n_samples, 2), permute, ds, unpermute, and recombine.
        dims = np.arange(X.ndim + 1)
        dims_to = list(np.roll(dims, 1))
        dims_from = list(np.roll(dims, -1))
        return torch.view_as_complex(fn_ds(torch.view_as_real(X).permute(*dims_to)).permute(*dims_from).contiguous())


def convolve(
    arr, 
    kernels, 
    take_abs=False, 
    fft_conv=False, 
    padding='same', 
    fast_length=False,
    DEVICE='cpu',
):
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
        fft_conv (bool):
            Whether to use FFT convolution.
        padding (str):
            Padding mode to use: \n
                * If fft_conv==False: ['valid', 'same'] \n
                * If fft_conv==True: ['full', 'valid', 'same'] \n
        fast_length (bool):
            Whether to use scipy.fftpack.next_fast_len to find the next fast
            length for the FFT.
        DEVICE (str):
            Device to use for computation.

    Returns:
        torch.Tensor:
            Result of the convolution.
    """
    assert all(isinstance(arg, torch.Tensor) for arg in [arr, kernels]), "arr and kernels should be torch tensors"

    arr = arr[None,:] if arr.ndim==1 else arr
    kernels = kernels[None,:] if kernels.ndim==1 else kernels

    arr = arr.to(DEVICE)[:,None,:]  ## Shape: (n_channels, 1, n_samples)
    kernels = kernels.to(DEVICE)  ## Shape: (n_kernels, win_size)
    
    if fft_conv:
        out = fftconvolve(
            x=arr,  
            y=(kernels)[None,:,:], 
            mode=padding,
            fast_length=fast_length,
        )
    else:
        flag_kernels_complex = kernels.is_complex()
        kernels = torch.flip(kernels, dims=[-1,])[:,None,:]  ## Flip because torch's conv1d uses cross-correlation, not convolution.
        
        if flag_kernels_complex:
            kernels = [torch.real(kernels), torch.imag(kernels)]
        else:
            kernels = [kernels,]

        out_conv = [torch.nn.functional.conv1d(
            input=arr, 
            weight=kernels, 
            padding=padding,
        ) for kernels in kernels]
        
        if flag_kernels_complex:
            out = torch.complex(*out_conv)
        else:
            out = out_conv[0]
        
    if take_abs:
        return torch.abs(out)
    else:
        return out


def fftconvolve(
    x, 
    y, 
    mode='valid',
    fast_length=False,
):
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

    Returns:
        torch.Tensor:
            Result of the convolution.
    """
    ## only if both are real, then use rfft
    if x.is_complex() == False and y.is_complex() == False:
        fft, ifft = torch.fft.rfft, torch.fft.irfft
    else:
        fft, ifft = torch.fft.fft, torch.fft.ifft
    
    ## Compute the convolution
    n_original = x.shape[-1] + y.shape[-1] - 1
    n = scipy.fftpack.next_fast_len(n_original) if fast_length else n_original
    
    f = fft(x, n=n, dim=-1) * fft(y, n=n, dim=-1)
    return apply_padding_mode(
        conv_result=ifft(f, n=n, dim=-1),
        x_length=x.shape[-1],
        y_length=y.shape[-1],
        mode=mode,
    )


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
            Result of the convolution with the specified padding mode.
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
    padding: str,
    downsample_factor: int,
    DEVICE_compute: str,
    DEVICE_return: str,
):
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
        DEVICE_compute (str):
            Device to use for computation.
        DEVICE_return (str):
            Device to use for returning the results.

    Returns:
        torch.Tensor:
            x-axis for the result of a convolution.
    """
    if downsample_factor == 1:
        DEVICE_compute = DEVICE_return

    ## If n_k is odd, then no offset, if even, then offset by 0.5
    ### PyTorch's conv1d and for 'same' pads more to the right, so on the first index of the output the kernel is centered at an offset of 0.5
    offset = 0.5 if n_k % 2 == 0 else 0

    x_axis_full = torch.arange(
        -(n_k-1)//2 + offset,
        n_s + (n_k-1)//2 + offset,
        dtype=torch.float32,
        device=DEVICE_compute,
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
        return_complex=False,
    ).squeeze().to(DEVICE_return)

    return x_axis