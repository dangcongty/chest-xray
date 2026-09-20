# Source parity notes

Reference snapshot: Ultralytics `6900c83b16eebee55c7b9de23b9ef447e6ff11e7` (2026-09-17).

## Architecture parity

The compact `YOLO26` module intentionally retains the same `model.<layer>...` state-dict hierarchy as the upstream detection model. Automated comparisons performed while building this release found:

- all upstream state-dict keys exist in the compact detector;
- all corresponding tensor shapes match;
- the compact model adds only one persistent buffer, `model.23.stride`;
- after copying the same weights, raw `boxes` and `scores` from both `one2many` and `one2one` branches have maximum absolute difference `0.0` on the tested input;
- parameter counts for COCO (`nc=80`) match the upstream YAML summaries:

| Scale | Parameters |
| --- | ---: |
| n | 2,572,280 |
| s | 10,009,784 |
| m | 21,896,248 |
| l | 26,299,704 |
| x | 58,993,368 |

The upstream parser silently changes the first two `C3k2` blocks to `C3k` internals for M/L/X. `model.py` preserves this scale-dependent behavior; omitting it produces superficially working models with incorrect parameter counts.

## Loss parity

With shared weights, input and targets, the compact dual-head loss matched the upstream three-element loss vector exactly in the comparison run:

- Task-Aligned Assignment with `topk=10` for one-to-many;
- STAL-style `topk=7`, reduced to one positive, for one-to-one;
- CIoU box loss;
- BCE classification loss;
- normalized L1 distance loss (the DFL-free YOLO26 meaning of the historical `dfl` gain);
- initial Progressive Loss weights `o2m=0.8`, `o2o=0.2`, decaying to `o2m=0.1`.

## Deliberate differences

The trainer is compact rather than a clone of the complete Ultralytics engine. It excludes multi-GPU/DDP, the full augmentation graph, rectangular batching, multi-scale training, automatic batch sizing, callbacks, exporters, integrations and all non-detection tasks. These differences can change a newly trained model's final accuracy even though the detector and loss primitives match.

