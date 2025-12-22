# Resolution study TODOs

## Other/Protocol
- [x] Identified potential problems with taps in the center (larger than all other taps, see slide 2)

## Ingestion & Preprocessing
- [x] Defined data structures for session metadata and data
- [x] Implemented data loading and dataset loading

## Event Segmentation
- [x] Data normalization (per channel with MAD)
- [ ] Implement "tap" detection on a single session, align with a protocol if provided. (Because I should be able to calculate expected tap times or frequencies)
- [ ] Save event table for later processing.

## Feature Extractions
- [ ] Extract per-channel peak amplitudes within the event window.
- [ ] Compute per-tap normalized vectors (L2 norm to control for tap strength if we don't have force sensor data)
  - [ ] Get channel fractions |A_c| / sum_c |A|
- [ ] Plot channel fractions vs. tap index.

## Interpretation
- [ ] Quantify pairwise discriminability between taps on different locations.
- [ ] Plot discriminability vs. spatial distance (dX)
- [ ] Define spatial resolution criterion.



