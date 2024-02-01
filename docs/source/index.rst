vqt
======

`Github Repository <https://github.com/richiehakim/vqt>`_


`vqt` is a simple-to-use Python package implementing the Variable Q-Transform.

--------
Installation
--------

From PyPI:

.. code-block:: bash

   pip install vqt

From source:

.. code-block:: bash

   git clone https://github.com/RichieHakim/vqt.git
   cd vqt
   pip install -e .

--------
Usage
--------

.. code-block:: python

   import vqt

   signal = X  ## numpy or torch array of shape (n_channels, n_samples)

   model = vqt.VQT(
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

   spectrograms, x_axis, frequencies = model(signal)




Indices and tables
------------------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _github: https://github.com/RichieHakim/vqt
