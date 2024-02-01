import torch
import numpy as np

import vqt

def test_torch_hilbert():
    # Test with a simple sinusoidal signal
    x = torch.sin(torch.linspace(0, 2 * np.pi, 100))
    xa = vqt.helpers.torch_hilbert(x)

    # Check if the output is complex
    assert xa.is_complex(), "Output of torch_hilbert should be complex"

    # Check if the imaginary part is non-zero (since Hilbert transform should produce a non-zero imaginary component for a real sinusoid)
    assert torch.any(torch.imag(xa) != 0), "Imaginary part of the output should be non-zero for a non-constant signal"


def test_make_VQT_filters():
    filters, freqs, wins = vqt.helpers.make_VQT_filters()

    # Check if filters, freqs, and wins are not None
    assert filters is not None, "Filters should not be None"
    assert freqs is not None, "Frequencies should not be None"
    assert wins is not None, "Window functions should not be None"

    # Check the shapes of the outputs
    assert len(filters.shape) == 2, "Filters should be a 2D tensor"
    assert len(freqs.shape) == 1, "Frequencies should be a 1D array"
    assert len(wins.shape) == 2, "Window functions should be a 2D tensor"


def test_VQT_initialization():
    vqt = vqt.VQT()
    assert vqt is not None, "VQT object should be initialized"

def test_VQT_call_with_simple_signal():
    vqt = vqt.VQT()
    x = torch.sin(torch.linspace(0, 2 * np.pi, 1000))
    spec, x_axis, freqs = vqt(x)

    # Check if outputs are not None
    assert spec is not None, "Spectrogram should not be None"
    assert x_axis is not None, "x-axis should not be None"
    assert freqs is not None, "Frequencies should not be None"

    # Check the shapes of the outputs
    assert len(spec.shape) == 3, "Spectrogram should be a 3D tensor"
    assert len(x_axis.shape) == 1, "x-axis should be a 1D tensor"
    assert len(freqs.shape) == 1, "Frequencies should be a 1D tensor"

def test_VQT_repr():
    vqt = vqt.VQT()
    repr_str = repr(vqt)
    assert isinstance(repr_str, str), "Representation should be a string"
