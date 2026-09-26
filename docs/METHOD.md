# Method and provenance

## Authoritative implementation

Original archive: `daoct_submission_v14_optional_layer_guard.zip`.
The `submission/` tree preserves its 24 files. Code V14 is historical scored
submission V5. Later V15/V16 experiments are not the third-place submission.

The top-level training wrapper is the authority for stage configuration, and
the inference wrapper is the authority for deployment settings. Some archived
comments describe historical local experiments, not final-test evidence.

## Training stages

| Stage | Default steps | Target images | Main mechanism |
|---|---:|---|---|
| Initial | 8000 | Unlabeled | FDA and EMA teacher consistency, confidence 0.7 |
| Refinement | 500 | Unlabeled | Pseudo-label CE, confidence 0.85, class balancing, axial boundary emphasis |
| FOV continuation | 250 | Not used for loss | Source-only continuous-ONH and FOV exit augmentation |

Stage 2 promotion requires source validation at least as high as the initial
student score by default. Stage 3 runs only after accepted refinement, tolerates
up to 0.01 source score drop by default and uses a 0.5 parameter interpolation
with the pre-FOV model when accepted. Failure/rejection retains earlier weights.
These safeguards monitor source performance; they do not guarantee target safety.

The model uses a SegResNet trunk (initial filters 16), instance normalization,
semantic segmentation plus auxiliary signed-distance and surface supervision.
The configured training loss includes Dice/CE, boundary and ordering terms.
Labels come only from the permitted source directory. The directory contract,
not automatic scanner/protocol recognition, enforces this separation.

## Domain robustness

Source augmentations include paired geometry, curvature, tilt, steep edges,
axial compression, black edges, crop/zoom and intensity changes. Lesion-like
intensity augmentation preserves labels; it is not a medically validated fluid
simulator or a disease segmentation head. Continuous-ONH augmentation is not
class-0 disc-removal supervision. The working grid is square, not native
aspect-ratio-preserving; lateral scaling and shape augmentation address some
geometry variation but do not remove this limitation.

FDA uses source/target Fourier amplitude mixing. The teacher is an exponential
moving average of the student, not an external foundation model. Target
pseudo-labels never constitute ground truth, and their errors can reinforce bias.

## Inference

Average probabilities from identity, horizontal flip and lateral scale 1.15.
Decode each A-scan with non-decreasing class IDs, allowing skipped classes.
Apply lateral despiking/median smoothing and empty-column correction; resize
to native dimensions and restore exactly black native columns to class 0.
If available, evaluate the pre-FOV anchor and use its prediction when class 2
is absent there but present in the candidate. This is an image-level heuristic.

## Evaluation interpretation

The bundled local metric combines per-class Dice and exp(-MASD/0.02), with
MASD normalized by image height. Aggregate challenge rankings must come from
official scoring reports, not a pooled average of selected examples.
Local decoder ablations are post-hoc evidence and do not measure isolated
component gains on the inaccessible final test set.
