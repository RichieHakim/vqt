## VQT: Variable Q-Transform
[![PyPI
version](https://badge.fury.io/py/vqt.svg)](https://badge.fury.io/py/vqt)

Contributions are welcome! Feel free to open an issue or a pull request.

### Variable Q-Transform

This is a novel python implementation of the variable Q-transform that was
developed due to the need for a more accurate and flexible VQT for the use in
research. It is battle-tested and has been used in a number of research
projects. <br>
- **Accuracy**: The approach is different in that it is a **direct
implementation** of a spectrogram  via a Hilbert transformation at each desired
frequency. This results in an exact computation of the spectrogram and is
appropriate for research applications where accuracy is critical. The
implementation seen in `librosa` and `nnAudio` uses recursive downsampling,
which can introduce artifacts in the spectrogram under certain conditions.
- **Flexibility**: The parameters and codebase are less complex than in other
libraries, and the filter bank is fully customizable and exposed to the user.
Built in plotting of the filter bank makes tuning the parameters easy and
intuitive.
- **Speed**: The backend is written using PyTorch, and allows for GPU
acceleration. It is faster than the `librosa` implementation under most cases,
and roughly as fast as the `nnAudio` implementation. See below section 'What to
improve on?' for more details on how to speed it up further.


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
it does not use the recursive downsampling algorithm from [this
paper](http://academics.wellesley.edu/Physics/brown/pubs/effalgV92P2698-P2701.pdf).
Instead, it computes the power at each frequency using either direct- or
FFT-convolution with a filter bank of complex oscillations, followed by a
Hilbert transform. This results in a more accurate computation of the same
spectrogram. The direct computation approach also results in code that is more
flexible, easier to understand, and it has fewer constraints on the input
parameters compared to `librosa` and `nnAudio`.

#### What to improve on?
Contributions are welcome! Feel free to open an issue or a pull request.

- Flexibility:
  - `librosa` parameter mode: It would be nice to have a mode that allows for
    the same parameters as `librosa` to be used.
  - Make `VQT` class a full `torch.nn.Module` so that it can be used in a
    `torch.nn.Sequential` model. Ensure backpropagation works.
  - Make `VQT` class compatible with `torch.jit.script` and `torch.jit.trace`.
  
- Speed:
  - **Lossless approaches**:
    - For the `fft_conv` approach: I believe a massive (5-100x) speedup is
      possible using a sparse or non-uniform FFT. A direct approach where only
      the non-zero frequencies are computed in the `fft`, product, and `ifft`
      should get us closer to a theoretically optimal lossless approach. There
      is an implmentation of the NUFFT in PyTorch
      [here](https://github.com/mmuckley/torchkbnufft).
    - For the `conv1d` approach: I think it would be much faster if we cropped
      the filters to remove the blank space from the higher frequency filters.
      This would be pretty easy to implement and could give a >10x speedup.
  - **Lossy approaches**:
    - Recursive downsampling: Under many circumstances (like when `Q_high` is
      not much greater than `Q_low`), recursive downsampling is fine.
      Implementing it would be nice just for completeness ([from this
      paper](http://academics.wellesley.edu/Physics/brown/pubs/effalgV92P2698-P2701.pdf))
    - For conv1d approach: Use a strided convolution.
    - For fftconv approach: Downsample using `n=n_samples_downsampled` in `ifft`
      function.
  - Non-trivial ideas that theoretically could speed things up:
    - An FFT implementation that allows for a reduced set of frequencies to be
      computed.

#### Demo:
<img src="docs/media/example_ECG.png" alt="ECG" width="500"  align="right"
style="margin-left: 10px"/>

```
import vqt
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy

data_ecg = torch.as_tensor(scipy.datasets.electrocardiogram()[:10000])
sample_rate = 360

my_vqt = vqt.VQT(
    Fs_sample=sample_rate,
    Q_lowF=2,
    Q_highF=8,
    F_min=1,
    F_max=120,
    n_freq_bins=150,
    win_size=1501,
    window_type='gaussian',
    downsample_factor=8,
    padding='same',
    fft_conv=True,
    take_abs=True,
    plot_pref=False,
)

specs, xaxis, freqs = my_vqt(data_ecg)

fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, )
axs[0].plot(np.arange(data_ecg.shape[0]) / sample_rate, data_ecg)
axs[0].title.set_text('Electrocardiogram')
axs[1].pcolor(
    xaxis / sample_rate, 
    np.arange(specs[0].shape[0]), specs[0] * (freqs)[:, None], 
    vmin=0, 
    vmax=30,
    cmap='hot',
)
axs[1].set_yticks(np.arange(specs.numpy()[0].shape[0])[::10], np.round(freqs.numpy()[::10], 1));
axs[1].set_xlim([13, 22])
axs[0].set_ylabel('mV')
axs[1].set_ylabel('frequency (Hz)')
axs[1].set_xlabel('time (s)')
plt.show()
```