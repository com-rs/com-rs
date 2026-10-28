Recordings
==========

Measured captures enter a link as recordings. :func:`comrs.read_sigmf` reads a
`SigMF <https://sigmf.org>`_ recording (the metadata ``.sigmf-meta`` and the samples
``.sigmf-data``) and returns the samples, the sampling rate and the metadata;
:func:`comrs.write_sigmf` writes one, in any of the SigMF sample formats:

.. code-block:: python

   import numpy as np
   import comrs

   x = np.exp(2j * np.pi * 0.01 * np.arange(1024))
   comrs.write_sigmf("capture", x, 1e6, "ci16_le", {"core:description": "a tone"})
   y, fs, meta = comrs.read_sigmf("capture")

In a specification, the block ``capture`` replaces the transmitter and the channel by
a recording (SigMF, raw IQ files or NPY arrays) and aligns the reference sequence on
it, so that the receiver DSP and the metrics run unchanged on measured data.
