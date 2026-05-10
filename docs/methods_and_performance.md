# Methods and Performance Draft

## Methods

### Study Task

We developed a lightweight vertebral landmark localization model to automatically identify T12-L5 vertebral body centers from volumetric CT scans. The immediate application was automated L3 slice selection for downstream body-composition analysis, while retaining multi-level vertebral localization outputs for quality control and future extension.

### Pseudo-label Generation and Dataset Curation

Initial vertebral pseudo-labels were generated from volumetric CT using automated vertebral segmentation masks. For each CT volume, T12-L5 vertebral masks were projected onto a sagittal representation, and the centroid of each available vertebral mask was used as the landmark target. Cases with inverted orientation, incomplete anatomic coverage, or visibly mismatched pseudo-labels were manually reviewed and excluded from the clean evaluation set.

The final clean training manifest contained 935 training cases, 186 validation cases, and 183 test cases. The clean test set included 1,097 vertebral landmarks after exclusion of reviewed bad labels and upper-abdomen-only scans unsuitable for complete T12-L5/L3 localization evaluation.

### Input Representation

Each 3D CT volume was first reoriented into canonical space. A sagittal slab centered on the estimated spine center was extracted using a half-width of 12 voxels. To preserve both skeletal and non-skeletal contextual information, three 2D channels were generated:

1. Soft-tissue mean projection, normalized with a `[-250, 500] HU` window.
2. Wide-HU maximum intensity projection, normalized with a `[-1000, 1500] HU` window.
3. Bone-focused maximum intensity projection, normalized with a `[150, 1200] HU` window and suppression of voxels below the lower bone threshold.

The resulting full-information sagittal image was resized to `384 x 256` pixels and used as a three-channel input.

### Model Architecture

The localization network was a compact 2D U-Net-like encoder-decoder with three downsampling stages, transposed-convolution upsampling, skip connections, and a final `1 x 1` convolutional prediction head. The network accepted a three-channel sagittal slab projection and produced six heatmaps corresponding to T12, L1, L2, L3, L4, and L5.

### Training

The model was trained using masked mean squared error between predicted heatmaps and Gaussian target heatmaps. Only present vertebrae contributed to the loss. Training used Adam optimization with an initial learning rate of `0.001`, batch size `8`, and a ReduceLROnPlateau scheduler. The best checkpoint was selected according to validation loss.

### Inference

At inference, each vertebral center was decoded as the maximum response location in the corresponding heatmap. Predicted 2D coordinates were mapped back to the canonical 3D CT index space. The L3 prediction was used to export an axial single-slice CT image for downstream L3-level analysis.

### Evaluation Metrics

Localization performance was assessed using Euclidean center-point error in pixels on the resized sagittal input space. Successful detection rate (SDR) was reported at 5, 10, 15, and 20 pixel thresholds:

`SDR@t = number of landmarks with localization error <= t pixels / total number of evaluated landmarks`

## Performance

On the curated clean test set, the model achieved a mean localization error of `2.85 px` and median error of `1.00 px` across all evaluated T12-L5 landmarks.

| Metric | Overall |
| --- | ---: |
| Test cases | 183 |
| Evaluated landmarks | 1,097 |
| Mean error | 2.85 px |
| Median error | 1.00 px |
| SDR@5px | 92.07% |
| SDR@10px | 93.16% |
| SDR@15px | 93.80% |
| SDR@20px | 94.53% |

Per-vertebra performance is summarized below.

| Vertebra | Mean error (px) | Median error (px) | SDR@5px | SDR@10px | SDR@15px | SDR@20px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T12 | 3.09 | 1.00 | 90.71% | 92.35% | 92.35% | 93.44% |
| L1 | 4.16 | 1.00 | 89.07% | 89.62% | 90.16% | 90.16% |
| L2 | 3.91 | 1.00 | 88.52% | 89.07% | 89.62% | 90.71% |
| L3 | 3.00 | 1.00 | 91.26% | 92.35% | 93.44% | 93.99% |
| L4 | 1.56 | 1.00 | 95.08% | 97.81% | 98.91% | 99.45% |
| L5 | 1.39 | 1.00 | 97.80% | 97.80% | 98.35% | 99.45% |

The L3-specific SDR@10px was `92.35%`, supporting the model's use for automated L3 slice selection.

## Recommended Figure Captions

**Figure 1. Overview of the lightweight L3 localization workflow.** A 3D CT volume is transformed into a full-information sagittal slab projection. A compact 2D heatmap network predicts T12-L5 vertebral centers, and the L3 prediction is mapped back to the CT volume to export the corresponding axial slice.

**Figure 2. Training curves.** Training and validation loss decreased rapidly during early epochs and plateaued after learning-rate reduction. The best model was selected by validation loss.

**Figure 3. Successful detection rate by vertebral level.** SDR@5px, SDR@10px, SDR@15px, and SDR@20px are shown for T12-L5 on the curated clean test set.

**Figure 4. Representative qualitative prediction.** Predicted T12-L5 vertebral centers are shown on a sagittal full-information projection, and the L3 prediction is mapped back to the volumetric CT to export the corresponding axial slice.
