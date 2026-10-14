# Data and references

Place the original assessment files in this directory. They are excluded from Git:

| File | Purpose |
|---|---|
| `train-test.csv` | 48,000 labeled loads, January–October 2025 |
| `validation.csv` | 12,000 unlabeled loads, November–December 2025 |
| `validation-predictions-template.csv` | Required validation IDs and prediction column |
| `december-chart-inputs.csv` | 31 fixed Lexington–Fort Wayne loads for December |

Original source files remain unchanged. Completed prediction files are written to `outputs/`.

## Included reference assets

- `city-coordinates.csv`: selected [GeoNames cities15000](https://download.geonames.org/export/dump/) city records, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), including IDs and source URLs.
- `routing-coordinate-overrides.csv`: the explicit routing-reference override.
- `city-boundaries-2025.geojson` and `us-state-outlines-2025.geojson`: US Census TIGER/Line boundaries used in the coordinate audit and static maps.
- `route_cache/`: the active HERE matrix snapshot and three example route geometries. Uncached requests require `HERE_API_KEY` in the environment; no key is stored in the repository.

## Assessment outputs

The supplied instructions request a GitHub repository with code and run instructions, validation predictions, a PDF/DOCX report with the split methodology and December chart, and a 2–3 minute Loom walkthrough. The assessment PDF referenced in the original notes is not present in this workspace.

The [root README](../README.md) contains the execution and scorer commands.
