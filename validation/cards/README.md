# Validation cards

A validation card states, for one algorithm, what it must give and why: the function, the
inputs, the expected outputs with their tolerance, and the exact source of the
expectation (a paper with its equation, table or figure; a standard with its section; a
reference implementation with its version). The cards are data, not code: `tests/test_cards.py`
runs every card of this folder against the Python interface of com-rs, and another library
can run the same cards by mapping `function` to its own functions.

```yaml
id: chang2002_fig4_floors                  # unique, the file name without .yaml
title: Error floors of Gray 16-QAM with an imperfect channel estimate
function: comrs.theory_ber_imperfect_csi   # the function under test
source:
  kind: paper                              # paper | standard | tool | closed_form
  reference: "M.-X. Chang, Y. T. Su, IEEE Trans. Wireless Commun. 1(4), 721-732, 2002"
  locator: "Fig. 4, read at Eb/N0 = 50 dB"
  file: 2002_chang_equalized-ofdm-rayleigh_twc.pdf   # in references/ (not distributed)
tolerance: {rel: 0.3}                      # default of the cases: rel, abs, or exact: true
note: "values read on the figure, to about 10 %"
cases:
  - inputs: {modulation: 16qam, ebn0_db: 50.0, rho1: 0.999}
    expected: 3.5e-3
```

- `inputs` are the keyword arguments of the call; a value `{construct: comrs.FiberParams, args: {...}}`
  is built by calling the named constructor with `args`.
- `expected` is a number, a list (compared element by element) or a recorded output of a
  reference tool, `{oracle: nr_ssb, path: ["0", "pss"]}` (`tests/oracle/<oracle>.json`).
- `output` (optional) picks a field of a returned dictionary or tuple (`gain`, `0`).
- `tolerance` passes when `|got - expected| <= abs + rel |expected|`, element by element;
  `exact: true` requires equality (bit-exact sequences).

Write the exponents of numbers with their sign (`2.14e+10`, `1.0e-3`): YAML 1.1, read by
PyYAML, takes `21.4e9` for a string. `schema.json` is the JSON Schema of a card.
