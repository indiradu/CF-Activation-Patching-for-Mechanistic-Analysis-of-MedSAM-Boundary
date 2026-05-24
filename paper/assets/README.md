# Paper Assets

This directory contains non-patient aggregate figures used by the paper source.

Patient-derived qualitative CXR panels are intentionally ignored by Git via:

```text
paper/assets/figure_qualitative_generators_*.png
```

The qualitative panel can be regenerated locally from restored data and
prediction manifests using:

```bash
python scripts/make_counterfactual_figure_panel.py --help
```

It is fine for the submitted PDF to contain the qualitative figure when allowed
by the relevant dataset licenses and review rules, but avoid pushing the image
panel to a public anonymous code repository unless redistribution is clearly
permitted.
