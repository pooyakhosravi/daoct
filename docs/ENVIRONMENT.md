# Environment qualification

The submitted source imports PyTorch, MONAI, NumPy, Pillow and SciPy. Some
optional paths also import OpenCV. The model module imports MONAI `MedNeXt`
even though the chosen architecture is SegResNet, so not every MONAI release
will import this snapshot successfully.

Before publishing runnable environment instructions:

1. Recover the exact challenge Docker image and digest from the final run.
2. Record Python, CUDA, PyTorch, MONAI and other installed package versions.
3. Build and test a fresh offline container with the original source.
4. Test source-only supervised loading and image-only target loading.
5. Exercise training promotion/rejection and native-size inference.
6. Publish a pinned environment and measured hardware/runtime, not estimates.

Syntax/provenance checks alone do not establish runtime reproducibility.
No Dockerfile or unverified version pins are supplied in this initial draft.

## Local smoke check, 2026-09-26

The selected model passed a CPU forward/backward test with synthetic 64 x 64
input. All four output tensors had the expected dimensions and finite values,
and the gradients were finite. Training and inference entry-point imports also
passed. The configured model has 1,574,212 parameters.

The existing Windows development environment reported:

| Component | Version |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.5.1+cu124 |
| MONAI | 1.5.1 |
| NumPy | 2.2.3 |
| Pillow | 11.3.0 |
| SciPy | 1.14.1 |

These are observed local package versions, not a lockfile or the secure-server
environment. The existing MONAI installation exports `MedNeXt`; a fresh install
of a package bearing the same version has not been qualified. The smoke check
does not exercise a complete training run or native-size file inference.
