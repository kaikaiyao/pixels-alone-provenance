# Transfer to a raw-pixel ResNet-18 victim

This experiment tests whether attacks learned through a CLIP surrogate transfer to a ResNet-18 that operates on RGB images. It fine-tunes an ImageNet-pretrained ResNet-18 on the real/diffusion benchmark, fits an MLP over frozen CLIP ViT-B/32 features to its scalar fake scores, and runs targeted PGD through the surrogate. The ResNet-18 receives the final attacked images for evaluation; all PGD steps use the surrogate.

Dependencies: Python 3.10+, PyTorch 2.3+ with a compatible torchvision 0.18+ and CUDA installation, NumPy, Pillow, scikit-learn, and open-clip-torch. The model libraries download pretrained ResNet-18 and CLIP weights on first use. Tests use pytest and run on CPU.

## Input

Provide a CSV with the following columns:

```text
benchmark_id,prompt_id,image_id,split,generator_id,seed,image_path
```

`prompt_id`, `image_id`, and `seed` are integers. Use distinct prompts for the `train`, `val`, and `test` splits. Generator IDs are `real`, `sd15`, `sdxl`, `sd21`, and `pixart`; `real` maps to binary label 0 and the four diffusion sources to label 1. The full benchmark has 600/300/300 prompts and one image per generator per prompt, giving 3,000/1,500/1,500 images. Name the full manifest `sample_manifest_full.csv` to enable the full-run configuration check.

Relative `image_path` entries are resolved against the manifest directory, or against `--image-root` when supplied. Absolute paths are also supported. Prepare the images using the [benchmark workflow](../../README.md#image-benchmark-and-interface-experiments), or use your own images with this manifest format.

## Run

From the repository root, after installing the dependencies:

```bash
python experiments/raw_pixel/run_experiment.py \
  --stage all \
  --subset full \
  --manifest data/sample_manifest_full.csv \
  --image-root data \
  --output-root outputs/raw-pixel \
  --run-id resnet18-transfer \
  --epochs 10 \
  --surrogate-epochs 100 \
  --attack-steps 40 \
  --attack-restarts 5 \
  --attack-samples-per-direction 128 \
  --save-examples-per-direction 0
```

The remaining defaults match the paper: seed `20260723`, training batch size 32, backbone/head learning rates `1e-4/1e-3`, weight decay `1e-4`, a 256-unit surrogate hidden layer, and query/attack perturbation budgets `8/255`. PGD uses step size `epsilon/4`. The surrogate uses GELU and LayerNorm and is fitted with soft-target binary cross-entropy; the lowest validation MSE selects its checkpoint. The query bank contains a clean and one uniformly perturbed variant of every training/validation image: 6,000 training and 3,000 validation replies for the full benchmark.

Victim training selects its checkpoint and binary threshold using validation data. The surrogate/attack stage requires a validation fake AUC of at least `0.85`; test metrics are reported separately. Attacks operate on the deterministic evaluation crop, `Resize(256) -> CenterCrop(224) -> ToTensor`. Results include attack success rates for all sources and initially correct sources, together with the attack-derived empirical gap estimate.

To run the stages separately, use `--stage train-victim` followed by `--stage surrogate-attack` with the same manifest, output root, run ID, and full settings. For a smaller run, use `--subset smoke` with a smaller manifest whose splits have distinct prompts; the script processes every row in the supplied manifest. Set `TORCH_HOME`, `HF_HOME`, and `PP_OPEN_CLIP_CACHE` to choose cache directories for downloaded weights.

Results are saved under `--output-root/--run-id`: model checkpoints, query features, metric JSON, per-image attack CSV, and run metadata. Set `--save-examples-per-direction` to a positive number to save attack examples.

## CPU tests

The tests check data handling and attack logic using synthetic inputs and a toy surrogate.

```bash
python -m pytest -q experiments/raw_pixel/tests
```
