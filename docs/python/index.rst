:html_theme.sidebar_secondary.remove:

comrs
=====

**Link-level simulation of digital communication systems, wireless and optical.**

``comrs`` is the Python interface of `com-rs <https://github.com/com-rs/com-rs>`_:
the blocks of a digital or optical link — sources, constellations, pulse shaping,
channels, fibre propagation, devices, receiver DSP and metrics — implemented in Rust
and driven from Python with NumPy arrays. Monte Carlo campaigns run on every core with
reproducible random streams, and their results come with confidence intervals.

.. code-block:: python

   import comrs

   rng = comrs.Rng(1)
   c = comrs.Constellation("qam", 16).normalized()
   bits = comrs.bits(4 * 100_000, rng)
   y = comrs.awgn_ebn0(c.modulate(bits), 10.0, 4, rng)
   comrs.ber(c.demodulate(y), bits), comrs.theory_ber(16, 10.0, "qam")

.. grid:: 1 2 2 2
   :gutter: 3
   :margin: 4 4 0 0

   .. grid-item-card:: Getting started
      :link: getting_started
      :link-type: doc

      Installation, conventions (arrays, generators, units) and a first link.

   .. grid-item-card:: Tutorials
      :link: auto_tutorials/index
      :link-type: doc

      Step by step: a first link, a classical single-carrier chain, a block of your
      own in Python, a campaign as a specification run from the command line, the
      profile of a chain, and the signals along it.

   .. grid-item-card:: User guide
      :link: user_guide/index
      :link-type: doc

      Monte Carlo campaigns in Python and on the parallel Rust runner, declarative
      specifications, recordings.

   .. grid-item-card:: Gallery of classical results
      :link: auto_examples/index
      :link-type: doc

      Error rates over AWGN and fading, information rates, pulse shaping, dispersion,
      nonlinear propagation, MIMO detection, space-time codes and capacity, PAPR
      reduction, amplifiers and predistortion — against theory, and against named
      figures and tables of the literature.

   .. grid-item-card:: Maturity
      :link: maturity
      :link-type: doc

      The evidence behind every module: validated against a publication, a standard
      or a reference implementation, against a closed form, or self-consistent only.

   .. grid-item-card:: Comparisons with other libraries
      :link: comparisons
      :link-type: doc

      OptiCommPy, GNPy, AFF3CT, Sionna, py3gpp, sigmf-python: what is compared, how
      closely, and the differences found.

   .. grid-item-card:: API reference
      :link: api/index
      :link-type: doc

      Every function and class, with the formula it implements.

.. toctree::
   :hidden:
   :maxdepth: 2

   getting_started
   auto_tutorials/index
   user_guide/index
   auto_examples/index
   performance
   comparisons
   maturity
   api/index
