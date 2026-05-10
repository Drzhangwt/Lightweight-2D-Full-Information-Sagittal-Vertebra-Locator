# Lightweight 2D Full-Information Sagittal Vertebra Locator

This repository provides a lightweight 2D vertebral landmark localization model for automatic T12-L5 center detection from 3D CT volumes. The model was designed for automated L3 slice selection before downstream body-composition analysis.

## Highlights

- Input: 3D CT volume, e.g. `.nii.gz`, `.nii`, `.nrrd`, `.mha`.
- Output: T12-L5 vertebral center coordinates, sagittal preview, and extracted L3 axial single-slice image.
- Model: compact 2D U-Net-like heatmap network.
- Input representation: 3-channel full-information sagittal slab projection.
- Clean test performance: overall SDR@10px `93.16%`; L3 SDR@10px `92.35%`.

## Repository Structure

```text
model/
  best_model.pt
src/
  vertebra_locator.py
results/
  training_summary.json
  training_history.json
  performance_sdr_v21_clean.json
  excluded_cases_v21.json
figures/
  figure_1_workflow.png
  figure_2_training_curves.png
  figure_3_sdr_by_vertebra.png
docs/
  methods_and_performance.md
examples/
  example_prediction_preview.png
```

## Installation

```bash
pip install -r requirements.txt
```

Install a PyTorch build that matches your local CUDA/CPU environment.

## Inference

```bash
python src/vertebra_locator.py \
  --input /path/to/ct_volume.nii.gz \
  --output-dir outputs/case001 \
  --device cuda:0 \
  --slice-format nrrd
```

Outputs:

- `vertebra_locations.json`
- `vertebra_locations.csv`
- `vertebra_locator_preview.png`
- `l3_single_slice.nrrd` or `l3_single_slice.nii.gz`

## Performance

Curated clean test set:

| Metric | Value |
| --- | ---: |
| Test cases | 183 |
| Landmarks | 1,097 |
| Mean error | 2.85 px |
| Median error | 1.00 px |
| SDR@5px | 92.07% |
| SDR@10px | 93.16% |
| SDR@15px | 93.80% |
| SDR@20px | 94.53% |

L3-specific performance:

| Metric | Value |
| --- | ---: |
| Mean error | 3.00 px |
| Median error | 1.00 px |
| SDR@5px | 91.26% |
| SDR@10px | 92.35% |
| SDR@15px | 93.44% |
| SDR@20px | 93.99% |

More details are available in [`docs/methods_and_performance.md`](docs/methods_and_performance.md).

## Example Figure

![Qualitative prediction example](figures/figure_4_qualitative_prediction_example.png)

## Notes

This repository is intended for academic research and reproducibility. It is not a regulatory-approved medical device and should not be used for clinical decision-making without independent validation.

## License

This repository uses a layered license:

- Source code: Apache License 2.0
- Model weights in `model/`: CC BY-NC 4.0, research and non-commercial use only
- Documentation, figures, and result summaries: CC BY 4.0

Commercial use, clinical product integration, or regulatory/medical-device deployment of the model weights requires explicit written permission.
