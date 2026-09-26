# DA-OCT: retinal layer segmentation across OCT devices

Code for our third-place solution in the DA-OCT Challenge. The model learns
retinal layer segmentation from labeled Topcon Maestro2 Macula 6x6 scans and
adapts using unlabeled scans from other devices and protocols.

This repository preserves the final submitted implementation, internally named
V14 (scored submission V5). Experimental variants are excluded.

## Final hidden-test results

| Evaluation | Official score |
|---|---:|
| Overall | 0.803378 |
| Macula | 0.840268 |
| Widefield | 0.766489 |

These values come from the organizer's final scoring report. They are distinct
from submission-phase leaderboard scores and local development results. The
third-place ranking was reported by the submitting author.

## Status

- `submission/` contains the original submitted code, without functional edits.
- No patient data, organizer plots, credentials, or model weights are included.
- Author list, copyright holders, exact server environment and trained weights
  require confirmation before public release.
- This is a private repository prepared for organizer coordination. MONAI Model
  Zoo packaging has not been completed.

## Method

The model learns from labeled Topcon Maestro2 Macula 6x6 images and uses other
training domains as unlabeled inputs. It requires no disease-status input.
A compact multi-head SegResNet predicts semantic labels, with auxiliary distance
and surface objectives during training. The deployed model starts without an
external pretrained trunk.

Training includes source image/mask augmentation, Fourier domain adaptation and
EMA teacher consistency. A second stage uses confident target pseudo-labels,
class balancing and axial boundary emphasis. Source validation gates promotion.
An optional supervised field-of-view continuation uses continuous-ONH and FOV
exit augmentation. Accepted continuation weights are averaged with the prior
checkpoint. This is parameter averaging, not prediction ensembling.

Inference uses percentile normalization, a 512-square working grid, horizontal
flip and lateral-scale probability averaging, ordered decoding, lateral surface
smoothing and empty-column masking. The optional pre-FOV anchor can replace a
prediction based on class-2 presence. Ordering allows skipped classes, but cannot
represent an internal background gap followed by deeper retinal labels.

See [method details](docs/METHOD.md) and [release checklist](docs/RELEASE.md).

## Repository contents

- `AGENTS.md`: onboarding, code map, data rules and contributor guidance.
- `submission/`: the 24 original challenge files, preserved byte for byte.
- `docs/`: method, provenance, environment and release notes.
- `tools/verify_submission.py`: dependency-free integrity and syntax checks.
- `tools/smoke_model.py`: optional CPU forward/backward check using synthetic input.

Verify the source snapshot from the repository root:

```sh
python tools/verify_submission.py
```

After preparing the runtime, check imports and the selected model:

```sh
python tools/smoke_model.py
```

The smoke check uses random tensors and does not measure segmentation quality.
See [provenance](docs/PROVENANCE.md) for the original archive hash.

## Run

Install a compatible GPU PyTorch build and the dependencies described in
[environment notes](docs/ENVIRONMENT.md). The exact server image is not yet
reconstructed in this draft. Do not treat this as a verified fresh-install recipe.

From `submission/`, train on a challenge-format training tree:

```sh
python train_test_monai_semi.py --data_root /path/to/train
```

The labeled directory must be `Topcon_Maestro2`; all other immediate device
directories are unlabeled. Do not mix labeled widefield or other-device scans
into that source directory. Images use `*-image.png`; source masks use matching
`*-mask.png` files with integer IDs 0-9.

Run inference with weights obtained from an authorized training run:

```sh
python infer_test_monai.py --input_dir /path/to/images --output_dir /path/to/predictions --model_path checkpoints/unet_maestro2_semi.pth
```

Keep `model_meta.json` and, when produced, `unet_maestro2_anchor.pth` beside the
main checkpoint. Predictions are native-size PNG masks with IDs 0-9.

The original challenge entry point is:

```sh
bash entrypoint.sh /path/to/input /path/to/output /absolute/path/to/submission
```

Use a writable submission directory. `main.py` skips training if it discovers
existing weights: use a clean copy for a new training run. Runtime model downloads
are not required. Dependencies must already be installed in the offline container.

## Evidence and limitations

Organizer-provided final-test figures support qualitative discussion only. The
underlying hidden images and masks are not available here. Earlier leaderboard
scores, local Iowa-derived proxies and synthetic release scores are different
evaluations and must not be presented as final-test performance.

This is research software, not a validated clinical device. No clinical outcome,
diagnostic accuracy or reliable fluid segmentation claim is made. There is no
dedicated fluid class or disease detector. No anatomical names are assigned to
class IDs here without the official mapping.

## License

BSD 2-Clause for project-owned code; see [LICENSE](LICENSE). Confirm copyright
holders and preserve applicable upstream notices before publication. Third-party
dependencies, data, figures and future weights retain their own applicable terms.
