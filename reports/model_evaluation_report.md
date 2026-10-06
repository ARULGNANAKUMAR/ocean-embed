# Model Evaluation Report — OceanEmbed SIH 26066

**Generated:** 2026-09-12T06:23:33.579099

## Architecture
- CNN Encoder + SpatialAttention + ResBlock Decoder + MC-Dropout

## Training
- epochs_run: 5
- best_val_loss: 0.298170268535614
- data_split: chronological 70/15/15

## ARGO Validation
- status: VALIDATED
- rmse: 2.9617
- mae: 2.6156
- bias: -0.6629
- correlation: 0.9142
- r2: 0.6948

## Uncertainty
- method: MC-Dropout (n=12)
- overall_confidence: 0.8657
- depth_uncertainty_max: 0.1905
