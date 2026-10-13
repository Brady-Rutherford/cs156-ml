"""On Cue: predict, from strictly causal class text and context, whether Brady raises his hand
in the next H seconds of a Minerva Forum class.

Labels come from a hue detector of the raised-hand icon on his tile in the class recording
(no face recognition). The design is `pipeline1/design/on_cue_spec.md` with the overrides in
`pipeline1/design/amendments.md`; constants live in `oncue.config`, table schemas in
`oncue.schemas`, and the synthetic test world in `oncue.synthetic`.

Run stages with `python -m oncue <subcommand>` (see `oncue.__main__`). Importing the package
is cheap: no stage module is imported until its subcommand runs.
"""

__version__ = "0.1.0"
