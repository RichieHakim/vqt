import torch
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from tqdm import tqdm

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
        DEVICE_compute='cpu',
        DEVICE_return='cpu',
        batch_size=1000,
        return_complex=False,
        filters=None,
        plot_pref=False,
        progressBar=True,
    ):
        """
        Variable Q Transform.
        Class for applying the variable Q transform to signals.

        This function works differently than the VQT from 
         librosa or nnAudio. This one does not use iterative
         lowpass filtering. Instead, it uses a fixed set of 
         filters, and a Hilbert transform to compute the analytic
         signal. It can then take the envelope and downsample.
        
        Uses Pytorch for GPU acceleration, and allows gradients
         to pass through.

        Q: quality factor; roughly corresponds to the number 
         of cycles in a filter. Here, Q is the number of cycles
         within 4 sigma (95%) of a gaussian window.

        RH 2022

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
                Whether to use a symmetric window or a single-sided window.
                - 'center': Use a symmetric / centered / 'two-sided' window.
                - 'left': Use a one-sided, left-half window. Only left half of the
                filter will be nonzero.
                - 'right': Use a one-sided, right-half window. Only right half of the
                filter will be nonzero.
            taper_asymmetric (bool):
                Only used if symmetry is not 'center'.
                Whether to taper the center of the window by multiplying center
                sample of window by 0.5.
            downsample_factor (int):
                Factor to downsample the signal by.
                If the length of the input signal is not
                 divisible by downsample_factor, the signal
                 will be zero-padded at the end so that it is.
            padding (str):
                Padding to use for the signal.
                'same' will pad the signal so that the output
                 signal is the same length as the input signal.
                'valid' will not pad the signal. So the output
                 signal will be shorter than the input signal.
            DEVICE_compute (str):
                Device to use for computation.
            DEVICE_return (str):
                Device to use for returning the results.
            batch_size (int):
                Number of signals to process at once.
                Use a smaller batch size if you run out of memory.
            return_complex (bool):
                Whether to return the complex version of 
                 the transform. If False, then returns the
                 absolute value (envelope) of the transform.
                downsample_factor must be 1 if this is True.
            filters (Torch tensor):
                Filters to use. If None, will make new filters.
                Should be complex sinusoids.
                shape: (n_freq_bins, win_size)
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
        self.Fs_sample, self.Q_lowF, self.Q_highF, self.F_min, self.F_max, self.n_freq_bins, self.win_size, self.downsample_factor, self.padding, self.DEVICE_compute, \
            self.DEVICE_return, self.batch_size, self.return_complex, self.plot_pref, self.progressBar = \
                Fs_sample, Q_lowF, Q_highF, F_min, F_max, n_freq_bins, win_size, downsample_factor, padding, DEVICE_compute, DEVICE_return, batch_size, return_complex, plot_pref, progressBar

    def _helper_ds(self, X: torch.Tensor, ds_factor: int=4, return_complex: bool=False):
        if ds_factor == 1:
            return X
        elif return_complex == False:
            return torch.nn.functional.avg_pool1d(X, kernel_size=[int(ds_factor)], stride=ds_factor, ceil_mode=True)
        elif return_complex == True:
            ## Unfortunately, torch.nn.functional.avg_pool1d does not support complex numbers. So we have to split it up.
            ### Split X, shape: (batch_size, n_freq_bins, n_samples) into real and imaginary parts, shape: (batch_size, n_freq_bins, n_samples, 2)
            Y = torch.view_as_real(X)
            ### Downsample each part separately, then stack them and make them complex again.
            Z = torch.view_as_complex(torch.stack([torch.nn.functional.avg_pool1d(y, kernel_size=[int(ds_factor)], stride=ds_factor, ceil_mode=True) for y in [Y[...,0], Y[...,1]]], dim=-1))
            return Z

    def _helper_conv(self, arr, filters, take_abs, DEVICE):
        out = torch.complex(
            torch.nn.functional.conv1d(input=arr.to(DEVICE)[:,None,:], weight=torch.real(filters.T).to(DEVICE).T[:,None,:], padding=self.padding),
            torch.nn.functional.conv1d(input=arr.to(DEVICE)[:,None,:], weight=-torch.imag(filters.T).to(DEVICE).T[:,None,:], padding=self.padding)
        )
        if take_abs:
            return torch.abs(out)
        else:
            return out

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
                Get units of time by dividing by self.Fs_sample.
            self.freqs (Torch tensor):
                Frequencies of the spectrogram.
        """
        if type(X) is not torch.Tensor:
            X = torch.as_tensor(X, dtype=torch.float32, device=self.DEVICE_compute)

        if X.ndim==1:
            X = X[None,:]

        ## Make iterator for batches
        batches = helpers.make_batches(X, batch_size=self.batch_size, length=X.shape[0])

        ## Make spectrograms
        specs = [self._helper_ds(
            X=self._helper_conv(
                arr=arr, 
                filters=self.filters, 
                take_abs=(self.return_complex==False),
                DEVICE=self.DEVICE_compute
                ), 
            ds_factor=self.downsample_factor,
            return_complex=self.return_complex,
            ).to(self.DEVICE_return) for arr in tqdm(batches, disable=(self.progressBar==False), leave=True, total=int(np.ceil(X.shape[0]/self.batch_size)))]
        specs = torch.cat(specs, dim=0)

        ## Make x_axis
        x_axis = torch.nn.functional.avg_pool1d(
            torch.nn.functional.conv1d(
                input=torch.arange(0, X.shape[-1], dtype=torch.float32)[None,None,:], 
                weight=torch.ones(1,1,self.filters.shape[-1], dtype=torch.float32) / self.filters.shape[-1], 
                padding=self.padding
            ),
            kernel_size=[int(self.downsample_factor)], 
            stride=self.downsample_factor, ceil_mode=True,
        ).squeeze()
        
        return specs, x_axis, self.freqs

    def __repr__(self):
        if self.using_custom_filters:
            return f"VQT with custom filters"
        else:
            return f"VQT object with parameters: Fs_sample={self.Fs_sample}, Q_lowF={self.Q_lowF}, Q_highF={self.Q_highF}, F_min={self.F_min}, F_max={self.F_max}, n_freq_bins={self.n_freq_bins}, win_size={self.win_size}, downsample_factor={self.downsample_factor}, DEVICE_compute={self.DEVICE_compute}, DEVICE_return={self.DEVICE_return}, batch_size={self.batch_size}, return_complex={self.return_complex}, plot_pref={self.plot_pref}"