import math
from typing import Union, List, Tuple, Dict, Any, Optional, Sequence, Iterable

import torch
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

def make_batches(
    iterable, 
    batch_size=None, 
    num_batches=None, 
    min_batch_size=0, 
    return_idx=False, 
    length=None,
    idx_start=0,
):
    """
    Make batches of data or any other iterable.
    RH 2021

    Args:
        iterable (iterable):
            iterable to be batched
        batch_size (int):
            size of each batch
            if None, then batch_size based on num_batches
        num_batches (int):
            number of batches to make
        min_batch_size (int):
            minimum size of each batch
        return_idx (bool):
            whether to return the slice indices of the batches.
            output will be [start, end] idx
        length (int):
            length of the iterable.
            if None, then length is len(iterable)
            This is useful if you want to make batches of 
             something that doesn't have a __len__ method.
        idx_start (int):
            starting index of the iterable.
    
    Returns:
        output (iterable):
            batches of iterable
    """

    if length is None:
        l = len(iterable)
    else:
        l = length
    
    if batch_size is None:
        batch_size = int(math.ceil(l / num_batches))
    
    for start in range(idx_start, l, batch_size):
        end = min(start + batch_size, l)
        if (end-start) < min_batch_size:
            break
        else:
            if return_idx:
                yield iterable[start:end], [start, end]
            else:
                yield iterable[start:end]


def gaussian(x=None, mu=0, sig=1):
    '''
    A gaussian function (normalized similarly to scipy's function)
    RH 2021
    
    Args:
        x (np.ndarray): 1-D array of the x-axis of the kernel
        mu (float): center position on x-axis
        sig (float): standard deviation (sigma) of gaussian
        
    Returns:
        gaus (np.ndarray): gaussian function (normalized) of x
        params_gaus (dict): dictionary containing the input params
    '''
    if x is None:
        x = np.linspace(-sig*5, sig*5, int(sig*7), endpoint=True)

    gaus = 1/(np.sqrt(2*np.pi)*sig)*np.exp((-((x-mu)/sig) **2)/2)

    return gaus


def make_scaled_wave_basis(
    mother, 
    lens_waves, 
    lens_windows=None, 
    interp_kind='cubic', 
    fill_value=0,
):
    """
    Generates a set of wavelet-like basis functions by scaling a mother wavelet
    to different sizes. \n
    
    Note that this does not necessarily result in a true
    orthogonal 'wavelet' basis set. This function uses interpolation to adjust
    the mother wavelet's size, making it suitable for creating filter banks with
    various frequency resolutions.

    RH 2024

    Parameters:
    - mother (np.ndarray): 
        A 1D numpy array representing the mother wavelet used as the basis for scaling.
    - lens_waves (int, list, tuple, np.ndarray): 
        The lengths of the output waves. Can be a single integer or a list/array of integers.
    - lens_windows (int, list, tuple, np.ndarray, None): 
        The window lengths for each of the output waves. If None, defaults to
        the values in lens_waves. Can be a single integer (applied to all waves)
        or a list/array of integers corresponding to each wave length.
    - interp_kind (str): 
        Specifies the kind of interpolation as a string ('linear', 'nearest',
        'zero', 'slinear', 'quadratic', 'cubic', where 'zero', 'slinear',
        'quadratic' and 'cubic' refer to a spline interpolation of zeroth,
        first, second or third order).
    - fill_value (float): 
        Value used to fill in for requested points outside of the domain of the
        x_mother. Can be anything from scipy.interpolate.interp1d. If not
        provided, defaults to 0.

    Returns:
        (tuple):
        - waves (list): 
            List of the scaled wavelets.
        - xs (list):
            List of the x-values for each of the scaled wavelets.

    Example:
    ```
    mother_wave = np.cos(np.linspace(-2*np.pi, 2*np.pi, 10000, endpoint=True))
    lens_waves = [50, 100, 200]
    lens_windows = [100, 200, 400]
    waves, xs = make_scaled_wave_basis(mother_wave, lens_waves, lens_windows)
    ```
    """
    assert isinstance(mother, np.ndarray), "mother must be a 1D array"
    assert mother.ndim == 1, "mother must be a 1D array"
    
    arraylikes = (list, tuple, np.ndarray)
    if isinstance(lens_waves, arraylikes):
        lens_waves = np.array(lens_waves, dtype=int)
        if lens_windows is None:
            lens_windows = lens_waves
        if isinstance(lens_windows, int):
            lens_windows = np.array([lens_windows] * len(lens_waves), dtype=int)
        if isinstance(lens_windows, arraylikes):
            assert len(lens_waves) == len(lens_windows), "lens_waves and lens_windows must have the same length"
            lens_windows = np.array(lens_windows, dtype=int)
        else:
            raise ValueError("lens_windows must be an int or an array-like")
    elif isinstance(lens_waves, int):
        if lens_windows is None:
            lens_windows = lens_waves
        if isinstance(lens_windows, int):
            lens_waves = np.array([lens_waves], dtype=int)
            lens_windows = np.array([lens_windows], dtype=int)
        if isinstance(lens_windows, arraylikes):
            lens_waves = np.array([lens_waves] * len(lens_windows), dtype=int)
            lens_windows = np.array(lens_windows, dtype=int)
        else:
            raise ValueError("lens_windows must be an int or an array-like")
    else:
        raise ValueError("lens_waves must be an int or an array-like")


    x_mother = np.linspace(start=0, stop=1, num=len(mother), endpoint=True) - 0.5

    interpolator = scipy.interpolate.interp1d(
        x=x_mother,
        y=mother, 
        kind=interp_kind, 
        fill_value=fill_value, 
        bounds_error=False, 
        assume_sorted=True,
    )

    waves = []
    xs = []
    for i_wave, (l_wave, l_window) in enumerate(zip(lens_waves, lens_windows)):
        x_wave = (np.linspace(start=0, stop=1, num=l_window, endpoint=True) - 0.5) * (l_window / l_wave)
        wave = interpolator(x_wave)
        waves.append(wave)
        xs.append(x_wave)

    return waves, xs


def torch_hilbert(x, N=None, dim=0):
    """
    Computes the analytic signal using the Hilbert transform.
    Based on scipy.signal.hilbert
    RH 2022
    
    Args:
        x (nd tensor):
            Signal data. Must be real.
        N (int):
            Number of Fourier components to use.
            If None, then N = x.shape[dim]
        dim (int):
            Dimension along which to do the transformation.
    
    Returns:
        xa (nd tensor):
            Analytic signal of input x along dim
    """
    assert x.is_complex() == False, "x should be real"
    n = x.shape[dim] if N is None else N
    assert n >= 0, "N must be non-negative"

    xf = torch.fft.fft(input=x, n=n, dim=dim)
    m = torch.zeros(n, dtype=xf.dtype, device=xf.device)
    if n % 2 == 0: ## then even
        m[0] = m[n//2] = 1
        m[1:n//2] = 2
    else:
        m[0] = 1 ## then odd
        m[1:(n+1)//2] = 2

    if x.ndim > 1:
        ind = [np.newaxis] * x.ndim
        ind[dim] = slice(None)
        m = m[tuple(ind)]

    return torch.fft.ifft(xf * m, dim=dim)


class VQT_filters:
    """
    A class for creating a set of filters for use in the VQT algorithm. \n
    Set Q_lowF and Q_highF to be the same value for a Constant Q Transform (CQT)
    filter set. Varying these values will varying the Q factor logarithmically
    across the frequency range. \n

    RH 2022, 2024

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
            Highest frequency to use (inclusive).
        n_freq_bins (int):
            Number of frequency bins to use.
        win_size (int, None):
            Size of the window to use, in samples. \n
            If None, will be set to the next odd number after Q_lowF * (Fs_sample / F_min).
        window_type (str, np.ndarray, list, tuple):
            Window type to use. \n
                * If string: Will be passed to scipy.signal.windows.get_window.
                  See that documentation for options. Except for 'gaussian',
                  which you should just pass the string 'gaussian' without any
                  other arguments.
                * If array-like: Will directly be used as the window for the low
                  frequency wavelet and will be scaled for higher frequencies.
        symmetry (str):
            Whether to use a symmetric window or a single-sided window.
            - 'center': Use a symmetric / centered / 'two-sided' window.
            - 'left': Use a one-sided, left-half window. Only left half of the
            filter will be nonzero.
            - 'right': Use a one-sided, right-half window. Only right half of the
            filter will be nonzero.
        taper_asymmetric (bool):
            Only used if symmetry is not 'center'. \n
            Whether to taper the center of the window by multiplying center
            sample of window by 0.5.
        mother_resolution (int):
            Resolution of the mother wavelet. Should be a large integer.
        plot_pref (bool):
            Whether to plot the filters.
    """
    def __init__(
        self,
        Fs_sample: int=1000,
        Q_lowF: int=3,
        Q_highF: int=20,
        F_min: int=10,
        F_max: int=400,
        n_freq_bins: int=55,
        win_size: Optional[int]=None,
        window_type: Union[str, np.ndarray, list, tuple]='gaussian',
        symmetry: str='center',
        taper_asymmetric: bool=True,
        mother_resolution: int=10000,
        plot_pref: bool=False,
    ):
        self.Fs_sample, self.Q_lowF, self.Q_highF, self.F_min, self.F_max, self.n_freq_bins, self.win_size, self.window_type, self.symmetry, self.taper_asymmetric, self.mother_resolution, self.plot_pref = \
            Fs_sample, Q_lowF, Q_highF, F_min, F_max, n_freq_bins, win_size, window_type, symmetry, taper_asymmetric, mother_resolution, plot_pref
        
    def make_filters(self,):
        """
        Returns:
            filters (Torch ndarray):
                Array of complex sinusoid filters.
                shape: (n_freq_bins, win_size)
            freqs (Torch array):
                Array of frequencies corresponding to the filters.
            wins (Torch ndarray):
                Array of window functions corresponding to each filter. \n
                shape: (n_freq_bins, win_size)
        """
        Fs_sample, Q_lowF, Q_highF, F_min, F_max, n_freq_bins, win_size, window_type, symmetry, taper_asymmetric, mother_resolution, plot_pref = \
            self.Fs_sample, self.Q_lowF, self.Q_highF, self.F_min, self.F_max, self.n_freq_bins, self.win_size, self.window_type, self.symmetry, self.taper_asymmetric, self.mother_resolution, self.plot_pref
        
        # if win_size % 2 != 1:
        #     raise ValueError("RH Error: win_size should be an odd integer")
        
        ## Make frequencies. Use a geometric spacing.
        freqs = np.geomspace(
            start=F_min,
            stop=F_max,
            num=n_freq_bins,
            endpoint=True,
            dtype=np.float32,
        )

        periods = 1 / freqs
        periods_inSamples = Fs_sample * periods

        if win_size is None:
            win_size = int(np.ceil(Q_lowF * (Fs_sample / F_min)))
            ## Make sure win_size is odd
            if win_size % 2 != 1:
                win_size += 1
            win_size = 3 if win_size < 3 else win_size

        ## Make windows
        if isinstance(window_type, str):
            ## Handle gaussian windows separately
            scales = np.geomspace(
                start=Q_lowF,
                stop=Q_highF,
                num=n_freq_bins,
                endpoint=True,
                dtype=np.float32,
            ) * periods_inSamples
            scales = np.clip(scales, a_min=1, a_max=None)
            if window_type == 'gaussian':
                ## Make sigmas for gaussian windows. Use a geometric spacing.
                window_type = ('gaussian', mother_resolution * 0.15)
            # else:
            ### Make mother wave
            mother_wave = scipy.signal.windows.get_window(window=window_type, Nx=mother_resolution, fftbins=False)
            
            wins, xs = make_scaled_wave_basis(mother_wave, lens_waves=scales, lens_windows=win_size)
            wins = torch.as_tensor(np.stack(wins, axis=0), dtype=torch.float32)

        elif isinstance(window_type, (np.ndarray, list, tuple)):
            mother_wave = np.array(window_type, dtype=np.float32)
        else:
            raise ValueError("window_type must be a string or an array-like")
            
        ### Make windows symmetric or asymmetric
        if symmetry=='center':
            pass
        else:
            heaviside = (torch.arange(win_size) <= win_size//2).float()
            if symmetry=='left':
                pass
            elif symmetry=='right':
                heaviside = torch.flip(heaviside, dims=[0])
            else:
                raise ValueError("symmetry must be 'center', 'left', or 'right'")
            wins *= heaviside
            ### Taper the center of the window by multiplying center sample of window by 0.5
            if taper_asymmetric:
                wins[:, win_size//2] = wins[:, win_size//2] * 0.5

        filts = torch.stack([torch.cos(torch.linspace(-np.pi, np.pi, win_size) * freq * (win_size/Fs_sample)) * win for freq, win in zip(freqs, wins)], dim=0)
        filts_complex = torch_hilbert(filts.T, dim=0).T
        ## Normalize filters to have unit magnitude
        filts_complex = filts_complex / torch.linalg.norm(filts_complex, ord=1, dim=1, keepdim=True)  ## Note: ord=1 is L1 norm. This makes the filters have unit magnitude.
        
        freqs = torch.as_tensor(freqs, dtype=torch.float32)

        if plot_pref:
            self._plot_VQT_filters(
                filts_complex=filts_complex,
                freqs=freqs,
                wins=wins,
                scales=scales,
                win_size=win_size,
                Fs_sample=Fs_sample,
            )

        self.filts_complex, self.freqs, self.wins, self.scales, self.win_size = \
            filts_complex, freqs, wins, scales, win_size

        return filts_complex, freqs, wins


    def _plot_VQT_filters(
        self,
        filts_complex: Optional[torch.Tensor]=None,
        freqs: Optional[torch.Tensor]=None,
        wins: Optional[torch.Tensor]=None,
        scales: Optional[np.ndarray]=None,
        win_size: Optional[int]=None,
        Fs_sample: Optional[int]=None,
    ):
        filts_complex = filts_complex if filts_complex is not None else self.filts_complex
        freqs = freqs if freqs is not None else self.freqs
        wins = wins if wins is not None else self.wins
        scales = scales if scales is not None else self.scales
        win_size = win_size if win_size is not None else self.win_size
        Fs_sample = Fs_sample if Fs_sample is not None else self.Fs_sample

        ## Plot
        fig_freqs, ax_freqs = plt.subplots()
        ax_freqs.plot(freqs)
        ax_freqs.set_xlabel('filter num')
        ax_freqs.set_ylabel('frequency (Hz)')

        fig_wins, ax_wins = plt.subplots()
        ax_wins.imshow(wins / torch.max(wins, 1, keepdims=True)[0], aspect='auto')
        ax_wins.set_ylabel('filter num')
        ax_wins.set_title('windows (gaussian)')

        fig_scales, ax_scales = plt.subplots()
        ax_scales.plot(scales)
        ax_scales.set_xlabel('filter num')
        ax_scales.set_ylabel('window width scales')    

        fig_filts, ax_filts = plt.subplots()
        ax_filts.imshow(torch.real(filts_complex) / torch.max(torch.real(filts_complex), 1, keepdims=True)[0], aspect='auto', cmap='bwr', vmin=-1, vmax=1)
        ax_filts.set_ylabel('filter num')
        ax_filts.set_title('filters (real component)')


        worN=win_size*4
        filts_freq = np.array([scipy.signal.freqz(
            b=filt,
            fs=Fs_sample,
            worN=worN,
        )[1] for filt in filts_complex])

        filts_freq_xAxis = scipy.signal.freqz(
            b=filts_complex[0],
            worN=worN,
            fs=Fs_sample
        )[0]

        fig_freqResponse, ax_freqResponse = plt.subplots()
        ax_freqResponse.plot(filts_freq_xAxis, np.abs(filts_freq.T));
        ax_freqResponse.set_xscale('log')
        ax_freqResponse.set_xlabel('frequency (Hz)')
        ax_freqResponse.set_ylabel('magnitude')

        plt.show()
        return {
            'fig_freqs': fig_freqs,
            'ax_freqs': ax_freqs,
            'fig_wins': fig_wins,
            'ax_wins': ax_wins,
            'fig_scales': fig_scales,
            'ax_scales': ax_scales,
            'fig_filts': fig_filts,
            'ax_filts': ax_filts,
            'fig_freqResponse': fig_freqResponse,
            'ax_freqResponse': ax_freqResponse,
        }