import copy
import math

import torch
import numpy as np

import vqt

    
import pytest
from hypothesis import given, strategies as st, settings, Verbosity, assume, example, seed, HealthCheck
import torch
import numpy as np
import scipy.signal

params_vqt = {
    'Fs_sample': 1000, 
    'Q_lowF': 3, 
    'Q_highF': 20, 
    'F_min': 10, 
    'F_max': 400, 
    'n_freq_bins': 55, 
    'win_size': None,
    'symmetry': 'center',
    'taper_asymmetric': True,
    'downsample_factor': 4,
    'padding': 'valid',
    'take_abs': True,
    'filters': None,
    'verbose': False,
    'plot_pref': False,
}


# 1. Test Zero Signal Transformation
def test_zero_signal_transformation():
    params = copy.deepcopy(params_vqt) 
    v = vqt.VQT(**params)  # Create a new instance for each test case
    input_signal = torch.zeros(1024, dtype=torch.float32)  # A zero signal of length 1024
    output, _, _ = v(input_signal)
    assert torch.all(output == 0), "VQT output for a zero signal should be zero across all frequency bins"

# 2. Test Impulse Signal Transformation
def test_impulse_signal_transformation():
    params = copy.deepcopy(params_vqt) 
    v = vqt.VQT(**params)  # Create a new instance for each test case
    input_signal = torch.zeros(1024, dtype=torch.float32)
    input_signal[512] = 1  # Set the middle point to 1, creating an impulse
    output, _, _ = v(input_signal)
    assert not torch.all(output == 0), "VQT output for an impulse signal should not be zero"
    assert output.shape[1] == v.filters.shape[0], "VQT output should have the same number of frequency bins as filters"

# 3. Test Sinusoidal Signal Transformation
@given(frequency=st.floats(
    # min_value=copy.deepcopy(params_vqt['F_min']), 
    # min_value=params_vqt['F_min'], 
    min_value=10,
    max_value=400,
    # max_value=copy.deepcopy(params_vqt['F_max']),
))
def test_peak_in_spectrogram_at_sine_wave_frequency(
    frequency,
):
    """Test to verify a peak in the spectrogram at a specific sine wave frequency."""
    params = copy.deepcopy(params_vqt) 
    v = vqt.VQT(**params)  # Create a new instance for each test case
    # Generate the sine wave signal
    t = torch.arange(0, 1.0, 1/params['Fs_sample'], dtype=torch.float32)
    input_signal = torch.sin(2 * np.pi * frequency * t)
    # Apply the VQT to this sine wave
    spectrogram, _, freqs = v(input_signal)
    # Convert freqs to a tensor for easier handling
    freqs_tensor = torch.as_tensor(freqs, dtype=torch.float32)
    # Locate the peak in the spectrogram
    peak_index = torch.argmax(spectrogram[0], dim=0)  # Assuming the output shape is (n_channels, n_freq_bins, time_bins) 
    assert torch.all(peak_index == peak_index[0]), "Expected a single peak in the spectrogram"
    peak_index = peak_index[0]
    # Find the frequency corresponding to the peak
    peak_frequency = freqs_tensor[peak_index]  # Taking the first peak if multiple time bins
    # Frequency nearest within the discrete frequency steps
    idx_nearest = torch.argmin(torch.abs(freqs_tensor - frequency))
    frequency_nearest = freqs_tensor[idx_nearest]
    # print(frequency, frequency_nearest.item(), peak_frequency.item(), peak_index.item(), idx_nearest.item())
    # Verify that the peak corresponds to the sine wave's frequency
    # assert torch.abs(peak_frequency - frequency_nearest) < frequency_tolerance, \
    #     f"Expected a peak at {frequency_nearest} Hz, found one at {peak_frequency.item()} Hz within a tolerance of {frequency_tolerance} Hz."
    assert np.abs(peak_index.item() - idx_nearest.item()) <= 1, \
        f"Expected a peak at index: {idx_nearest.item()} and frequency: {frequency_nearest.item()} Hz, found one at index: {peak_index.item()} and frequency: {peak_frequency.item()} Hz within a tolerance of 2 indices."
        
# 4. Test Constant Signal Transformation
def test_constant_signal_transformation():
    params = copy.deepcopy(params_vqt) 
    v = vqt.VQT(**params)  # Create a new instance for each test case
    input_signal = torch.ones(1024, dtype=torch.float32)  # A constant signal
    output, _, _ = v(input_signal)
    # In a constant signal, the energy should be concentrated at the lowest frequency bin
    assert torch.all(output[0] > output[1:]), "VQT output for a constant signal should peak at the lowest frequency bin"

# 5. Test various params
@settings(max_examples=1000, deadline=None, suppress_health_check=(HealthCheck.too_slow,))
@given(
    Fs_sample=st.floats(min_value=1, max_value=10000),
    Q_lowF=st.floats(min_value=1, max_value=100),
    Q_highF=st.floats(min_value=1, max_value=100),
    F_min=st.floats(min_value=1, max_value=1000),
    F_max=st.floats(min_value=1, max_value=1000),
    n_freq_bins=st.integers(min_value=1, max_value=100),
    win_size=st.integers(min_value=1, max_value=1000) | st.none(),
    symmetry=st.sampled_from(['center', 'left', 'right']),
    taper_asymmetric=st.booleans(),
    downsample_factor=st.integers(min_value=1, max_value=100),
    padding=st.sampled_from(['valid', 'same']),
    # filters=st.none() | st.sampled_from([None]),
    # plot_pref=st.booleans(),

    n_channels=st.integers(min_value=1, max_value=100),
    n_dim=st.integers(min_value=1, max_value=2),
)
def test_vqt_params(
    Fs_sample,
    Q_lowF,
    Q_highF,
    F_min,
    F_max,
    n_freq_bins,
    win_size,
    symmetry,
    taper_asymmetric,
    downsample_factor,
    padding,
    # take_abs,
    # filters,
    # plot_pref,

    n_channels,
    n_dim,
):
    ## Clip win_size to be at most 1001 and just set it 
    if win_size is not None:
        win_size = min(win_size, 1001)
    else:
        win_size = min(int(math.ceil(Q_lowF * (Fs_sample / F_min))), 1001)
    params = {
        'Fs_sample': Fs_sample,
        'Q_lowF': Q_lowF,
        'Q_highF': Q_highF,
        'F_min': F_min,
        'F_max': F_max,
        'n_freq_bins': n_freq_bins,
        'win_size': win_size,
        'symmetry': symmetry,
        'taper_asymmetric': taper_asymmetric,
        'downsample_factor': downsample_factor,
        'padding': padding,
        # 'take_abs': take_abs,
        # 'filters': filters,
        # 'plot_pref': plot_pref,
    }
    params.update({k: v for k, v in params_vqt.items() if k not in params})
    v = vqt.VQT(**params)

    # Make signal
    len_signal = params['win_size'] if params['win_size'] is not None else int(params['Q_lowF'] * params['Fs_sample'])
    input_signal = torch.rand(n_channels, len_signal + np.random.randint(1, 1000), dtype=torch.float32)
    if n_dim == 1:
        input_signal = input_signal[0]
    # Apply the VQT to this signal
    output, _, _ = v(input_signal)
    
    # Perform validations on the output
    assert output is not None, "Output should not be None"
    # Check output shape
    assert output.shape[1] == params['n_freq_bins'], "VQT output shape does not match the number of frequency bins"
    # Check all output is real if take_abs is True
    if params['take_abs']:
        assert torch.all(torch.isreal(output)), "VQT output should be real if take_abs is True"


def test_vqt_filters():
    params = copy.deepcopy(params_vqt) 
    # Generate the filters
    filters, freqs, wins = vqt.helpers.make_VQT_filters(
        Fs_sample=params['Fs_sample'],
        Q_lowF=params['Q_lowF'],
        Q_highF=params['Q_highF'],
        F_min=params['F_min'],
        F_max=params['F_max'],
        n_freq_bins=params['n_freq_bins'],
        win_size=params['win_size'],
        symmetry=params['symmetry'],
        taper_asymmetric=params['taper_asymmetric'],
        plot_pref=params['plot_pref']
    )
    filters, freqs, wins = filters.numpy(), freqs.numpy(), wins.numpy()

    ## Filters, freqs, and wins should all have the following properties:
    ### They are not all zeros, not all ones, and do not contain NaNs or infinities
    ### Filters should additionally have shape (n_freq_bins, win_size)
    if params['win_size'] is not None:
        assert filters.shape == (params['n_freq_bins'], params['win_size']), "VQT filters have an unexpected shape"
    else:
        assert filters.shape[0] == params['n_freq_bins'], "VQT filters have an unexpected shape"
    fns = {
        'not_all_zeros': lambda x: ~np.all(x == 0),
        'not_all_ones': lambda x: ~np.all(x == 1),
        'no_nans': lambda x: ~np.any(np.isnan(x)),
        'no_infs': lambda x: ~np.any(np.isinf(x)),
    }
    args = {
        'filters': filters,
        'freqs': freqs,
        'wins': wins,
    }
    for fn_name, fn in fns.items():
        for arg_name, arg in args.items():
            assert fn(arg), f"Test failed: arg: {arg_name} failed test: {fn_name}"
    
    ## Check that the filters are complex
    assert np.iscomplexobj(filters), "VQT filters are not complex"


#####################
## Other functions ##
#####################


# Test Hilbert Transform Equivalence
def test_hilbert_transform_equivalence():
    params = copy.deepcopy(params_vqt) 
    v = vqt.VQT(**params)  # Create a new instance for each test case
    # Generate a random signal
    input_signal = np.random.rand(1024).astype(np.float64)  ## Implementations for fft are different between numpy and torch for float32
    # Use the VQT class's Hilbert transform
    hilbert_vqt = vqt.helpers.torch_hilbert(torch.as_tensor(input_signal)).numpy()
    # Compare with scipy's Hilbert transform
    hilbert_scipy = scipy.signal.hilbert(input_signal)
    # Check if they are approximately equal
    assert np.allclose(hilbert_vqt, hilbert_scipy, rtol=1e-6), "Hilbert transforms do not match"
 

# Test convolution functions
def test_conv_accuracy():
    conditions = {
        'x': [
            ('ones_6', np.ones(6).astype(np.float32)),
            ('zeros_1center', np.array([0, 0, 0, 1, 0, 0]).astype(np.float32)),
        ],
        'k': [
            ('ones_3', np.ones(3).astype(np.float32)),
            ('ones_4', np.ones(4).astype(np.float32)),
            ('mid_1', np.array([0, 1, 0]).astype(np.float32)),
            ('mid_1_off', np.array([0, 0, 1, 0]).astype(np.float32)),
        ],
        'padding': ['valid', 'same'],
        'real': [False, True],
        'take_abs': [False, True],
        'fft': [False, True],
    }

    results = {}  # Dictionary to store results

    # Iterate through all combinations of conditions
    for x_name, x in conditions['x']:
        for k_name, k in conditions['k']:
            for padding in conditions['padding']:
                for take_abs in conditions['take_abs']:
                    for real in conditions['real']:
                        k_tu = scipy.signal.hilbert(k) if not real else copy.deepcopy(k)
                        for fft in conditions['fft']:
                            y_vqt = vqt.vqt.convolve(
                                torch.as_tensor(x),
                                torch.as_tensor(k_tu),
                                padding=padding, 
                                fast_length=True,
                                take_abs=take_abs,
                                fft_conv=fft
                            ).numpy()[0][0]
                            y_np = np.convolve(x, k_tu, mode=padding)
                            y_np = np.abs(y_np) if take_abs else y_np
                            key = (x_name, k_name, padding, take_abs, real, fft)
                            results[key] = {'y_vqt': np.round(y_vqt, 2), 'y_np': np.round(y_np, 2)}
            
        # Assertions and comparisons
        for key in results:
            x_name, k_name, padding, take_abs, real, fft = key
            # Compare FFT True vs FFT False results for each condition except fft flag
            if fft:  # Only check when fft is True, comparing it to fft False
                key_no_fft = (x_name, k_name, padding, take_abs, real, False)
                close = np.allclose(results[key]['y_vqt'], results[key_no_fft]['y_vqt'], rtol=1e-5)
                if not close:
                    print(f"Comparing {key} with FFT False")
                    print(f"result: {close}")
                    print(results[key]['y_vqt'])
                    print(results[key_no_fft]['y_vqt'])
                    print()
                assert close, f"Convolution result for {key} does not match FFT False"
            
            # Compare to NumPy's result when padding is 'valid'
            if padding == 'valid':
                close = np.allclose(results[key]['y_vqt'], results[key]['y_np'], rtol=1e-5)
                if not close:
                    print(f"Comparing {key} with NumPy's convolution result")
                    print(f"result: {close}")
                    print(results[key]['y_vqt'])
                    print(results[key]['y_np'])
                    print()
                assert close, f"Convolution result for {key} does not match NumPy's"

            # Compare to known outputs when when padding is 'same'
            if padding == 'same' and real == True:
                if x_name == 'ones_6' and k_name == 'ones_4':
                    close = np.allclose(results[key]['y_vqt'], np.array([3, 4, 4, 4, 3, 2]), rtol=1e-5)
                    if not close:
                        print(f"Comparing {key} with known output")
                        print(f"result: {close}")
                        print(results[key]['y_vqt'])
                        print(np.array([3, 4, 4, 4, 3, 2]))
                    assert close, f"Convolution result for {key} does not match expected output"