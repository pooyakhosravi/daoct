# Agent and contributor guide

This file applies to the whole repository. Start here when investigating the
method, updating documentation, or preparing an experiment.

## Project purpose and release status

This repository contains the final DA-OCT Challenge submission for retinal
layer segmentation across devices and acquisition protocols. The solution
placed third, as reported by the submitting author. The final official scores
are in `README.md`.

The submitted implementation is internally called V14 and corresponds to scored
submission V5 in the development history. Later experiments are not part of this
snapshot. The repository contains code and documentation, without datasets,
organizer figures, pretrained models or trained challenge weights.

The repository is private pending organizer coordination. Do not change its
visibility, publish weights, or create a public release without an explicit
request. BSD 2-Clause covers project-owned code. Upstream attribution and release
tasks are documented in `docs/PROVENANCE.md` and `docs/RELEASE.md`.

## First steps

1. Read `README.md`, `docs/METHOD.md`, and `docs/ENVIRONMENT.md`.
2. Run `git status --short` and preserve any existing contributor changes.
3. Run the dependency-free source check from the repository root:

   ```sh
   python tools/verify_submission.py
   ```

4. In a prepared scientific Python environment, run:

   ```sh
   python tools/smoke_model.py
   ```

The first command checks the original source hashes, Python syntax and local
imports. The second imports the entry points, builds the selected model and
checks output shapes and finite gradients using random CPU tensors. Neither
command establishes segmentation quality or reproduces full challenge training.

There is no qualified fresh-install recipe, lockfile or Dockerfile yet. The
environment document records a successful local smoke test and its package
versions. The code imports `MedNeXt` from MONAI even when using SegResNet. A MONAI
installation missing that export will fail at import time. Do not silently patch
the archive or claim that an untested dependency pin reproduces the server.

## Preserve the submitted snapshot

`submission/` contains all 24 original files, byte for byte. Their hashes and
sizes are recorded in `docs/submission_manifest.json`. Git attributes prevent
line-ending conversion in this tree. CI runs `tools/verify_submission.py`.

- Keep documentation and tooling changes outside `submission/` when possible.
- Do not run formatters, rename modules, clean up old comments, or remove unused
  branches in the submitted tree as incidental work.
- Do not regenerate the manifest just to make a failing integrity check pass.
- For an explicitly requested implementation change, agree on a separate
  experimental tree or branch and document how it differs from the submission.
  Preserve access to the original snapshot and its manifest.
- Function definitions and comments can describe unused experiments. Trace the
  wrappers' selected configuration before claiming a feature was deployed.

## Data rules

Only Topcon Maestro2 Macula 6x6 labels may supervise training. Images from other
devices and Topcon widefield protocols may be used without their labels.
The loader enforces this through directory selection, not image recognition.

Expected training layout:

```text
train/
  Topcon_Maestro2/              # Labeled macula 6x6 only
    Healthy/
      example-image.png
      example-mask.png
    Diseased/
      example-image.png
      example-mask.png
  Topcon_Maestro2_unlabeled/    # Includes eligible unlabeled Topcon scans
    ...-image.png
  Heidelberg_Spectralis/
    ...-image.png
  Zeiss_Cirrus/
    ...-image.png
```

`Topcon_Maestro2` is the labeled directory name. Other immediate device
directories are selected as unlabeled inputs. Never place labeled widefield or
other-device data inside the labeled directory. Masks use integer IDs 0 through
9 and match their image dimensions. Do not infer anatomical class names without
the official mapping. Prediction requires no disease-status input.

The organizer's OCT5k-derived sample data is a programmatic blueprint. Its
simulated textures do not reproduce actual scanner styles. AI-READI references
with Iowa-initialized, manually edited boundaries were local development tools.
They are not hidden-test ground truth and are not bundled here. Keep evaluation
subjects and their reference masks out of adaptation and model training.

## Code map

| File or directory | Responsibility |
|---|---|
| `submission/entrypoint.sh` | Challenge shell entry point with input, output and submission directory arguments |
| `submission/main.py` | Finds existing weights, trains if absent, then starts inference |
| `submission/train_test_monai_semi.py` | Authoritative configuration, preprocessing, training stages, acceptance checks and checkpoint blending |
| `submission/preresize.py` | Pre-resizes the training tree for cached loading |
| `submission/infer_test_monai.py` | Authoritative deployed prediction settings and optional anchor loading |
| `submission/daoct/data/dataset.py` | Supervised/unlabeled datasets, image and mask loading, normalization and caching |
| `submission/daoct/data/transforms.py` | Paired geometry and image appearance transformations, including experimental operators |
| `submission/daoct/models/build.py` | Model factory and parameter count |
| `submission/daoct/models/multihead.py` | Shared trunk, segmentation/distance/surface heads and label-derived auxiliary targets |
| `submission/daoct/models/load.py` | Checkpoint loading, architecture metadata and segmentation output extraction |
| `submission/daoct/engine/train.py` | Training loop, validation, losses, teacher updates and checkpoint selection |
| `submission/daoct/engine/infer.py` | Preprocessing, view averaging, decoding, native-size output and class-presence fallback |
| `submission/daoct/da/fda.py` | Fourier amplitude adaptation using unlabeled images |
| `submission/daoct/da/mean_teacher.py` | Exponential-moving-average teacher and consistency/pseudo-label objectives |
| `submission/daoct/da/surface_refine.py` | Ordered dynamic-programming decoding, lateral smoothing and empty-column handling |
| `submission/daoct/losses/seg_losses.py` | Segmentation, boundary and ordering losses |
| `submission/daoct/eval/metrics.py` | Local Dice and normalized surface-distance score |
| `tools/` | Release verification and synthetic model smoke check |

`submission/daoct/data/` is Python source. Keep it tracked. The root `.gitignore`
uses `/data/` for datasets specifically to avoid hiding this package.

## Method overview

The selected network is a compact MONAI SegResNet with initial filters 16,
instance normalization and 1,574,212 parameters. A shared trunk feeds retinal
class logits, normalized distance-to-boundary predictions and per-A-scan surface
positions. The distance target in `sdm_target` is nonnegative distance to the
nearest boundary, despite historical use of the abbreviation SDM.

All auxiliary training targets derive from labeled masks. The selected model
starts without an external pretrained trunk. MIRAGE and other external-model
ensembles were separate experiments and are excluded from this solution.

The training wrapper runs these stages:

1. Initial training: 8,000 steps, labeled macula supervision, geometry and
   appearance augmentation, Fourier adaptation and EMA teacher consistency.
2. Target refinement: 500 steps, confident pseudo-labels with threshold 0.85,
   class balancing and emphasis on interfaces detected along each A-scan.
   Fourier adaptation is disabled for this stage. An acceptance check compares
   performance on labeled macular validation data with the initial student.
3. Optional geometry continuation: 250 steps after accepted refinement, using
   labeled macula data with continuous disc-like and field-of-view exit
   transformations. Accepted weights are blended 50:50 with preceding weights.

The wrapper exposes `DAOCT_STEPS`, `DAOCT_REFINE_STEPS`, `DAOCT_REFINE_MIN_DELTA`,
`DAOCT_FOV_STEPS`, `DAOCT_FOV_MAX_SOURCE_DROP`, and `DAOCT_FOV_SOUP_ALPHA`.
Inspect their uses before overriding them. Shortened runs change the experiment
and do not reproduce final training. Acceptance checks on macular data cannot
guarantee performance on other devices.

Inference uses a square 512 x 512 working grid, percentile normalization and
probability averaging over the original, horizontally flipped and mildly wider
views. Ordered decoding permits zero-thickness layers but prevents an internal
background gap followed by deeper retinal labels. Lateral smoothing and
empty-column handling follow decoding. Outputs return to native dimensions,
with exactly black cutoff columns assigned class 0.

An optional pre-continuation anchor can replace the candidate's full prediction
when class 2 is absent in the anchor prediction but present in the candidate.
Keep the anchor and model metadata with the main weights when they are produced.

## Running the code

Use an authorized dataset and a writable working copy. From `submission/`:

```sh
python train_test_monai_semi.py --data_root /path/to/train
python infer_test_monai.py --input_dir /path/to/images --output_dir /path/to/predictions --model_path checkpoints/unet_maestro2_semi.pth
```

Training writes under `submission/checkpoints/`, including preprocessed data.
The deployed checkpoint is `unet_maestro2_semi.pth`; its companion files are
`model_meta.json` and, when produced, `unet_maestro2_anchor.pth`. The output masks
are PNGs with class IDs 0-9 at the input image dimensions.

The original container entry point is:

```sh
bash entrypoint.sh /path/to/input /path/to/output /absolute/path/to/submission
```

`main.py` uses `input/train` for training when present. It selects `input/val/images`
or `input/testing_data` for prediction when available. It searches several
checkpoint names and skips training if it finds existing weights. For a new
training run, use a fresh writable copy without checkpoints. Do not delete
someone else's weights to force retraining.

Challenge runs are offline. Dependencies must be installed beforehand. Avoid
adding runtime downloads. Full training is a substantial GPU job: do not launch
it for a documentation edit or without confirming the requested scope and data.

## Testing and experiment reporting

- Run the source verification after repository or packaging changes.
- Run the synthetic model check after environment or model-related work.
- For intentional runtime changes, add targeted tests for the affected loader,
  transformation, loss, decoder or checkpoint path in the experimental version.
- Record the baseline, data split, random seeds, configuration, training budget
  and checkpoint lineage for comparisons. Separate training, adaptation and
  evaluation subjects. Keep target labels evaluation-only.
- Report macula, widefield and other-device results separately. Distinguish
  local references, synthetic checks, submission-phase scores and final scores.
- Do not attribute a combined experiment's gain to one component without a
  matched comparison. Small local gains may reflect sampling variation.
- Describe the tests actually run and any gaps. A smoke check is not an
  end-to-end training/inference test or evidence of clinical validity.

## Publication and safety boundaries

Never commit patient scans, masks, DICOM metadata, organizer plots, credentials,
checkpoint files, cached embeddings or large experiment artifacts as incidental
changes. Review staged files before pushing. Ignore rules are a safeguard, not
proof that all staged files are safe to publish.

Public-release follow-up includes author/copyright confirmation, upstream terms,
the exact server environment, authorized final weights and organizer coordination.
MONAI Model Zoo packaging remains pending. Local proxy-trained weights must not
be described as final challenge weights. This is research software, without a
validated clinical indication or dedicated fluid segmentation output.
