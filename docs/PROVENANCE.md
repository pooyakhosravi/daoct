# Submitted source provenance

The `submission/` directory is a byte-for-byte copy of the 24 files in
`daoct_submission_v14_optional_layer_guard.zip`.

Archive SHA-256:

```text
331efc00b3f9e499cd144589fcd1d6b203b68f2b6a6db9d47b64425a6dd81583
```

`submission_manifest.json` records each file's SHA-256 and byte count. Run
`python tools/verify_submission.py` to check the snapshot without installing
training dependencies. CI runs the same check. Git attributes preserve original
source bytes across Windows and Linux checkouts.

Documentation, repository configuration and verification tools were added for
this release. The submitted training and inference code was not modified.

The final scores were transcribed from the organizer-provided final
`scores.json`: overall 0.8033784165500384, macula 0.8402682047325621,
widefield 0.7664886283675145. The final report, organizer plots and underlying
images are excluded from this repository.

## Upstream attribution

The challenge entry-point conventions and baseline pipeline derive from the
[official DA-OCT baseline](https://github.com/wusmai/miccai-challenge-daoct-baseline).
The local baseline checkout has no top-level license file. Confirm the
starting-kit terms and any required notices with the organizers before public
release. The BSD 2-Clause license in this repository covers project-owned code.
It does not replace third-party terms.

The implementation imports PyTorch and MONAI, including MONAI SegResNet, and
uses NumPy, Pillow and SciPy. Optional paths use OpenCV and torchvision.
Dependencies are not vendored. MIRAGE code, pretrained models and cached
embeddings are not included in the final submission or this repository.
