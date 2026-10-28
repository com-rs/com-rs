Getting started
===============

Installation
------------

``comrs`` is built from source with `maturin <https://www.maturin.rs>`_ and a Rust
toolchain (1.85 or later):

.. code-block:: bash

   git clone https://github.com/com-rs/com-rs
   cd com-rs
   python -m venv .venv && source .venv/bin/activate
   pip install maturin numpy
   maturin develop --release

The development dependencies are dependency groups of ``pyproject.toml`` (pip 25.1 or
later): ``pip install --group test`` for the tests (they read the recorded outputs of
the oracles), ``pip install --group lint`` for ruff, ``pip install --group docs`` for
this documentation. The package is typed (``py.typed`` and a stub of the extension).

Conventions
-----------

Arrays
   Signals are 1-D ``complex128`` arrays (array-likes are converted), bits are
   ``uint8`` arrays of 0 and 1, symbol indices ``int64`` arrays. Several modes
   (polarisations) are 2-D arrays ``(modes, samples)``.

Randomness
   There is no global generator: every function that draws takes a :class:`comrs.Rng`
   and advances it, so a simulation is reproducible from its seed. The generator is the
   one of the Rust crates — the same seed gives the same draws in Rust and in Python —
   and :meth:`comrs.Rng.stream` derives the independent generator of every trial of a
   Monte Carlo campaign.

Units and names
   Physical quantities are in SI units unless the name says otherwise: ``ebn0_db``,
   ``esn0_db``, ``snr_db``, ``power_dbm``, ``length_km``, ``alpha_db_km``,
   ``d_ps_nm_km``… :math:`E_b/N_0` and :math:`E_s/N_0` are always distinguished by
   name. ``fs`` is a sampling rate, ``fc_hz`` a carrier frequency, ``noise_var`` a noise
   variance :math:`E|w|^2`; MIMO matrices are ``(n_rx, n_tx)``. The tuning parameters
   are keyword-only (``comrs.pulse_shape("rrc", 4, rolloff=0.25)``).

Closed forms
   The ``theory_*`` functions and the conversions take a scalar (and return a float) or
   an array-like (and return an array of the same shape):
   ``comrs.theory_ber(16, np.arange(0, 12))``.

Errors
   Invalid parameters raise :class:`ValueError`, numerical failures
   :class:`ArithmeticError`.

Conventions of the blocks
   Gray labels, pulse grids, metrics and fibre models follow
   `OptiCommPy <https://github.com/edsonportosilva/OptiCommPy>`_, which serves as the
   test oracle: a user of one finds the same blocks under the same names in the other.

A first link
------------

16-QAM over AWGN, decided symbol by symbol, against the closed-form bit error rate:

.. code-block:: python

   import comrs

   rng = comrs.Rng(2024)
   c = comrs.Constellation("qam", 16).normalized()   # Gray labels, unit energy

   bits = comrs.bits(4 * 200_000, rng)                # 4 bits per symbol
   x = c.modulate(bits)
   y = comrs.awgn_ebn0(x, 10.0, 4, rng)               # Eb/N0 = 10 dB, 4 bits/symbol

   print(comrs.ber(c.demodulate(y), bits))            # 1.80e-3
   print(comrs.theory_ber(16, 10.0, "qam"))           # 1.75e-3

The :doc:`gallery <auto_examples/index>` turns such snippets into the classical curves,
and the :doc:`user guide <user_guide/index>` shows how to run them as Monte Carlo
campaigns with stopping rules and confidence intervals.
