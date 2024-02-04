## VQT: Variable Q-Transform
[![PyPI
version](https://badge.fury.io/py/vqt.svg)](https://badge.fury.io/py/vqt)

## Call for contributors!
I am looking for contributors to help improve this library. If you are
interested, please make a pull request or contact me directly.

### Variable Q-Transform

This is a novel python implementation of the variable Q-transform that was
developed due to the need for a more accurate and flexible VQT for the use in
research. <br>
- Accuracy: The approach is different in that it is a **direct implementation**
of a spectrogram  via a Hilbert transformation at each desired frequency. This
results in an exact computation of the spectrogram and is appropriate for
research applications where accuracy is critical. The implementation seen in
`librosa` and `nnAudio` uses recursive downsampling, which can introduce
artifacts in the spectrogram under certain conditions.
- Flexibility: The parameters and codebase are less complex than in other
libraries, and the filter bank is fully customizable and exposed to the user.
Built in plotting of the filter bank makes tuning the parameters easy and
intuitive.
- Speed: The backend is written using PyTorch, and allows for GPU acceleration
and backpropagation. In most cases it is faster than the `librosa`
implementation, especially when using a GPU. Also, when the
`downsample`/`hop_length` parameter is low (<32), it is as fast or faster than
the `nnAudio` implementation.


### Installation
From PyPI: `pip install vqt`

From source:
```
git clone https://github.com/RichieHakim/vqt.git
cd vqt
pip install -e .
```

**Requirements**: `torch`, `numpy`, `scipy`, `matplotlib`, `tqdm` <br>
These will be installed automatically if you install from PyPI.
  
### Usage
<img src="docs/media/filter_bank.png" alt="filter_bank" width="300"
align="right"  style="margin-left: 10px"/>

```
import vqt

signal = X  ## numpy or torch array of shape (n_channels, n_samples)

transformer = vqt.VQT(
    Fs_sample=1000,  ## In Hz
    Q_lowF=3,  ## In periods per octave
    Q_highF=20,  ## In periods per octave
    F_min=10,  ## In Hz
    F_max=400,  ## In Hz
    n_freq_bins=55,  ## Number of frequency bins
    DEVICE_compute='cpu',
    return_complex=False,
    filters=None,  ## Use custom filters
    plot_pref=False,  ## Can show the filter bank
)

spectrograms, x_axis, frequencies = transformer(signal)
```
<img src="docs/media/freqs.png" alt="freqs" width="300"  align="right"
style="margin-left: 10px"/>

#### What is the Variable Q-Transform?

The Variable Q-Transform (VQT) is a time-frequency analysis tool that generates
spectrograms, similar to the Short-time Fourier Transform (STFT). It can also be
defined as a special case of a wavelet transform, as well as the generalization
of the Constant Q-Transform (CQT). In fact, the VQT subsumes the CQT and STFT as
both can be recreated using specific parameters of the VQT.

#### Why use the VQT?

It provides enough knobs to tune the time-frequency resolution trade-off to suit
your needs.

#### How exactly does this implementation differ from others?
<img src="docs/media/freq_response.png" alt="freq_response" width="300"
align="right"  style="margin-left: 10px"/>

This function works differently than the VQT from `librosa` or `nnAudio` in that
it does not use recursive downsampling. Instead, it uses a fixed set of filters,
and a Hilbert transform to compute the analytic signal. It can then take the
envelope and downsample. This results in a more accurate computation of the
spectrogram. The tradeoff is that under certain conditions, it can be slower
than the recursive downsampling approach, but usually not by much. The direct
computation approach also results in code that is more flexible, easier to
understand, and it has fewer constraints on the input parameters compared to
`librosa` and `nnAudio`.

#### What to improve on?

- Flexibility:
  - Window function: Currently, the window function is hard-coded to be a
    Gaussian. This should be made flexible and should ideally include at least
    the Hamming window.
  - `librosa` parameter mode: It would be nice to have a mode that allows for
    the same parameters as `librosa` to be used.
  
- Speed:
  - Currently, it is likely that the existing code is about as fast as it can be
    without sacrificing accuracy, flexibility, or code clarity. All the
    important operations are done in PyTorch (with backends in `C` or `CUDA`).
  - Allowing for some loss in accuracy:
    - For conv1d approach: Use a strided convolution.
    - For fftconv approach: Downsampl using `n=n_samples_downsampled` in `ifft`
      function.
  - Non-trivial ideas that theoretically could speed things up:
    - An FFT implementation that allows for a reduced set of frequencies to be
      computed.
    - A


#### Demo:
<img src="docs/media/example_ECG.png" alt="ECG" width="500"  align="right"
style="margin-left: 10px"/>

```
import vqt
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy

data_ecg = scipy.datasets.electrocardiogram()

transformer = vqt.VQT(
    Fs_sample=360,
    Q_lowF=3,
    Q_highF=20,
    F_min=1,
    F_max=180,
    n_freq_bins=55,
    win_size=1501,
    downsample_factor=8,
    padding='same',
    return_complex=False,
    plot_pref=True,
    progressBar=False,
)

specs, xaxis, freqs = transformer(data_ecg)

fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, )
axs[0].plot(data_ecg)
axs[0].title.set_text('Electrocardiogram')
axs[1].pcolor(xaxis, np.arange(specs[0].shape[0]), specs[0] * (freqs)[:, None])
axs[1].set_yticks(np.arange(specs[0].shape[0])[::5], np.round(freqs[::5], 1));
axs[1].set_xlim([43000, 48000])
axs[0].set_ylabel('mV')
axs[1].set_ylabel('frequency (Hz)')
axs[1].set_xlabel('time (s)')
plt.show()
```