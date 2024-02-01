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
        batch_size = np.int64(np.ceil(l / num_batches))
    
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
    if n % 2: ## then even
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


def make_VQT_filters(    
    Fs_sample=1000,
    Q_lowF=3,
    Q_highF=20,
    F_min=10,
    F_max=400,
    n_freq_bins=55,
    win_size=501,
    symmetry='center',
    taper_asymmetric=True,
    plot_pref=False
):
    """
    Creates a set of filters for use in the VQT algorithm.

    Set Q_lowF and Q_highF to be the same value for a 
     Constant Q Transform (CQT) filter set.
    Varying these values will varying the Q factor 
     logarithmically across the frequency range.

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
            Highest frequency to use (inclusive).
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
        plot_pref (bool):
            Whether to plot the filters.

    Returns:
        filters (Torch ndarray):
            Array of complex sinusoid filters.
            shape: (n_freq_bins, win_size)
        freqs (Torch array):
            Array of frequencies corresponding to the filters.
        wins (Torch ndarray):
            Array of window functions (gaussians)
             corresponding to each filter.
            shape: (n_freq_bins, win_size)
    """

    assert win_size%2==1, "RH Error: win_size should be an odd integer"
    
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

    ## Make sigmas for gaussian windows. Use a geometric spacing.
    sigma_all = np.geomspace(
        start=Q_lowF,
        stop=Q_highF,
        num=n_freq_bins,
        endpoint=True,
        dtype=np.float32,
    )
    sigma_all = sigma_all * periods_inSamples / 4

    ## Make windows
    ### Make windows gaussian
    wins = torch.stack([gaussian(torch.arange(-win_size//2, win_size//2), 0, sig=sigma) for sigma in sigma_all])
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
    filts_complex = filts_complex / torch.sum(torch.abs(filts_complex), dim=1, keepdims=True)
    
    ## Plot
    if plot_pref:
        plt.figure()
        plt.plot(freqs)
        plt.xlabel('filter num')
        plt.ylabel('frequency (Hz)')

        plt.figure()
        plt.imshow(wins / torch.max(wins, 1, keepdims=True)[0], aspect='auto')
        plt.ylabel('filter num')
        plt.title('windows (gaussian)')

        plt.figure()
        plt.plot(sigma_all)
        plt.xlabel('filter num')
        plt.ylabel('window width (sigma of gaussian)')    

        plt.figure()
        plt.imshow(torch.real(filts_complex) / torch.max(torch.real(filts_complex), 1, keepdims=True)[0], aspect='auto', cmap='bwr', vmin=-1, vmax=1)
        plt.ylabel('filter num')
        plt.title('filters (real component)')


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

        plt.figure()
        plt.plot(filts_freq_xAxis, np.abs(filts_freq.T));
        plt.xscale('log')
        plt.xlabel('frequency (Hz)')
        plt.ylabel('magnitude')

    return filts_complex, freqs, wins