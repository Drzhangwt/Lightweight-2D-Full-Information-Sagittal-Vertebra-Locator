# Model Card

## Model

Lightweight 2D full-information sagittal vertebra locator for T12-L5 landmark localization.

## Intended Use

Research use for automated vertebral landmark localization and L3 slice selection from volumetric CT scans.

## Not Intended For

The model is not a regulatory-approved medical device and should not be used for clinical diagnosis or treatment decisions without independent validation.

Commercial use, clinical product integration, or regulatory/medical-device deployment of the model weights is not permitted without explicit written permission.

## Inputs

3D CT volumes readable by SimpleITK, including NIfTI, NRRD, MHA/MHD, and similar formats.

## Outputs

- T12-L5 vertebral center coordinates.
- Sagittal preview image.
- L3 single-slice axial CT image.

## Training Summary

- Training cases: 935
- Validation cases: 186
- Clean test cases: 183
- Input channels: 3
- Architecture: compact 2D U-Net-like heatmap network
- Optimizer: Adam
- Batch size: 8
- Initial learning rate: 0.001

## Clean Test Performance

- Mean error: 2.85 px
- Median error: 1.00 px
- SDR@5px: 92.07%
- SDR@10px: 93.16%
- SDR@15px: 93.80%
- SDR@20px: 94.53%

## Limitations

Performance may degrade on scans with incomplete lumbar coverage, severe anatomic distortion, unusual acquisition orientation, metal artifacts, or distributions substantially different from the training cohort.

## License

- Source code: Apache License 2.0
- Model weights: CC BY-NC 4.0, research and non-commercial use only
- Documentation and figures: CC BY 4.0
