# Validation cards

A validation card is one comparison of com-rs with its source, written as data: what is
compared, with which paper, standard or tool (and its exact equation, table or figure),
the expected values with their tolerance and the reason for that tolerance, the
differences found, and how to produce the measured values again. Reading the card is
enough to know what was checked; rerunning it does not need the paper:

```text
just card tal2015_fig1            # = python tools/cards.py run tal2015_fig1
python tools/cards.py list        # every card, its runner and tier
```

```yaml
id: tal2015_fig1                           # unique, the file name without .yaml
title: Word error rate of SC and list decoding of a (2048, 1024) polar code
literature: tal2015                        # its entry of validation/literature.yaml
source:
  kind: paper                              # paper | standard | tool | closed_form
  reference: "I. Tal, A. Vardy, List decoding of polar codes, IEEE Trans. Inf. Theory 61(5), 2213-2226, 2015"
  locator: "Fig. 1, read at 1.5, 2 and 2.5 dB"
  file: coding/2015_tal_list-decoding-polar-codes_it.pdf   # in references/<theme>/ (not distributed)
description: >                             # the chain, in a few lines
  A polar code of length 2048 and rate 1/2 ... decoded by SC and lists of 8 and 32.
run:                                       # how the measured values are produced (one of)
  test: validation/tests/tal2015.rs::tal_vardy_2015_fig_1
tier: campaign                             # ci (every run of the tests) | campaign (by hand)
duration: "130 s on an Apple M6"
tolerance: {factor: 1.6}                   # default of the cases
why: "the construction differs from the paper's ..."      # why this tolerance
findings:                                  # differences with the source, explained or not
  - "SC sits about 0.07 dB to the right of the figure, from the other construction"
cases:
  - {inputs: {list: 1, ebn0_db: 1.5}, expected: 0.26}
  - inputs: {claim: list_gain_at_2db}
    expected: {min: 10}                    # a range: the claims of the text
    note: "WER of SC over that of the list of 8 at 2 dB: 14 on the figure"
```

## Runners

- `function: comrs.<name>`: a function of the Python interface, called with the `inputs`
  of every case as keyword arguments (`tests/test_cards.py`, every run of `pytest`). A
  value `{construct: comrs.FiberParams, args: {...}}` is built by calling the named
  constructor with `args`; `output` picks a field of a returned dictionary or tuple;
  `expected` may be a recorded output of a reference tool, `{oracle: nr_ssb, path: ["0",
  "pss"]}` (`tests/oracle/<oracle>.json`). Another library can run the same cards by
  mapping the function to its own.
- `test: validation/tests/<file>.rs::<name>`: a Rust test that only measures. It loads
  the card (`com_validation::card::Card::load("<id>")`), records each measured value
  for the case of the same `inputs` (`card.check(&json!({...}), value)`) and finishes
  (`card.finish()`), which prints the table of the comparison and fails if a case failed
  or was never measured. When several tests measure parts of one card, `run` names the
  file and each test keeps its part (`Card::part(&json!({"config": "A"}))`).
- `spec: specs/validation/<name>.yaml`: a specification verified by `comrs verify`; its
  `expect` list holds the curves, tolerances and `why`, and the card has no cases.

## Rules

- A case passes when `|got - expected| <= atol + rtol |expected|`, or `expected/f <= got
  <= f expected` with `factor: f` (error rates over several decades), or, for `expected:
  {min: a, max: b}`, when `a <= got <= b` (either bound may be left out); `exact: true`
  requires equality (bit-exact sequences). These are the rules of `comrs verify`
  (`com_spec::verify::Tolerance`).
- Every tolerance that is not exact says `why` (on the card or the case): the precision
  of the published value (printed digits, a figure read to 10 %), the sampling error, or,
  when the tolerance is only the agreement obtained, that it is so.
- `findings` records the differences with the source: misprints, gaps explained by a
  different method, and gaps not explained.
- Write the exponents of numbers with their sign (`2.14e+10`, `1.0e-3`): YAML 1.1, read by
  PyYAML, takes `21.4e9` for a string. `schema.json` is the JSON Schema of a card;
  `python tools/cards.py --check` (in `just check`) checks that the runner exists and
  loads the card, that the literature entry exists and that the file is in a theme
  folder of `references/`.
