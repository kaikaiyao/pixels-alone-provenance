<h1 align="center">[NeurIPS '26] Can Pixels Alone Reveal Image Origin?</h1>
<h3 align="center">Minimax Limits and Learnable Interfaces<br>for Passive Provenance</h3>
<p align="center">Kai Yao<br>School of Informatics, University of Edinburgh</p>
<p align="center"><a href="https://arxiv.org/abs/2609.30997"><strong>Paper</strong></a> &nbsp; | &nbsp; <a href="#overview">Overview</a> &nbsp; | &nbsp; <a href="#main-findings">Findings</a> &nbsp; | &nbsp; <a href="#installation">Get started</a> &nbsp; | &nbsp; <a href="#citation">Citation</a></p>

---

## Overview

We study how reliably an image's origin can be inferred from its pixels after an attacker edits it. This includes distinguishing real from AI-generated images and identifying a particular generator.

The paper separates a statistical limit from the vulnerability of a particular verifier. We prove that the best robust target-acceptance gap equals the minimum total variation distance between the target distribution and the distributions reachable by editing source images. We then show how learning a verifier's public responses enables targeted attacks. Given an emulator with uniform error ε on the attack region, an attack optimized through it achieves target acceptance within 2ε plus optimization error of the white-box optimum. For logistic and softmax heads over public features, we derive conditions for exact and stable recovery from score responses.

## Main findings

Our experiments check the theory in a finite setting and test attacks on real and diffusion-generated images.

| Evaluation | Finding |
|:--|:--|
| Finite-state minimax identity | The two population LP optima differ by at most 4.44 × 10⁻¹⁶ across 20 settings. |
| Finite-state estimation | Mean absolute error decreases from 0.02094 at n = 250 to 0.00451 at n = 5,000. |
| Controlled public CLIP verifier | All four evaluated source-to-target directions reach a targeted attack success rate (ASR) of 100% at an L∞ budget of 8/255. |
| ResNet-18 victim, fake → real | 45/117 initially correct sources succeed under the tested 8/255 transfer attack. |
| ResNet-18 victim, real → fake | 0/122 initially correct sources succeed under the same protocol. |
| Paired quality evaluation | At 2/255, 45/50 initially correct attack instances succeed; at 8/255, 50/50 succeed. |

## Contents

The repository includes the verifier and attack code, benchmark preparation tools, and the finite-state, ResNet-18 transfer, and image-quality experiments.

| Experiment | What it runs | Code |
|:--|:--|:--|
| Finite-state study | Independent transport and verifier LPs, plus estimation from sampled distributions | [Guide](experiments/barrier_i_finite_state/README.md) |
| Head recovery | Recovery of logistic and softmax heads from score responses | [`exact_recovery.py`](src/passive_provenance/pipelines/exact_recovery.py) |
| Verifier interfaces | Emulation and attacks using full scores, labels, binary feedback, and abstention | [`interfaces.py`](src/passive_provenance/interfaces.py), [`stages`](src/passive_provenance/stages) |
| ResNet-18 transfer | Attacks through a CLIP surrogate against an RGB ResNet-18 | [Guide](experiments/raw_pixel/README.md) |
| Image quality | Paired measurements of clean and attacked images | [`perceptual_audit.py`](src/passive_provenance/stages/perceptual_audit.py) |

## Installation

Use Python 3.10 or newer. Install a compatible PyTorch and torchvision pair for your machine using the [official PyTorch instructions](https://pytorch.org/get-started/locally/), then run:

```bash
git clone https://github.com/kaikaiyao/pixels-alone-provenance.git
cd pixels-alone-provenance
python -m pip install -e ".[test]"
python -m pytest tests -q
```

For image generation, install `python -m pip install -e ".[generation]"`. For LPIPS evaluation, install `python -m pip install -e ".[perceptual]"`. Pretrained models download their weights on first use.

The [finite-state study](experiments/barrier_i_finite_state/README.md) runs on CPU with NumPy and SciPy.

## Image benchmark and interface experiments

[`configs/default.yaml`](configs/default.yaml) specifies the prompt-disjoint train/validation/test split, models, interfaces, query caps, and attack parameters. B0 contains real, Stable Diffusion 1.5, and SDXL images. B1 adds Stable Diffusion 2.1 and PixArt.

Set `data.coco_root` to a local COCO 2017 directory, or set `PP_COCO_ROOT`; set `PP_ARTIFACTS_ROOT` if generated data and outputs should be stored outside the checkout. The COCO directory should contain `val2017/` and `annotations/captions_val2017.json`, as read by [`datasets.py`](src/passive_provenance/datasets.py).

Run stages from the repository root. For example, to prepare and generate the full B0 benchmark:

```bash
export PP_COCO_ROOT=/path/to/coco2017
export PP_ARTIFACTS_ROOT=/path/to/experiment_outputs
python -m passive_provenance.stages.prepare_benchmark --config configs/default.yaml --benchmark b0
python -m passive_provenance.stages.generate_benchmark --config configs/default.yaml --benchmark b0 --model sd15 --subset full
python -m passive_provenance.stages.generate_benchmark --config configs/default.yaml --benchmark b0 --model sdxl --subset full
```

Then train the verifier, fit an emulator to its responses, and run the attack:

```bash
python -m passive_provenance.stages.extract_features --config configs/default.yaml --benchmark b0 --subset full
python -m passive_provenance.stages.train_verifier --config configs/default.yaml --verifier-id d1_softmax --subset full
python -m passive_provenance.stages.eval_clean --config configs/default.yaml --verifier-id d1_softmax --subset full
python -m passive_provenance.stages.collect_interface_queries --config configs/default.yaml --verifier-id d1_softmax --interface-id full_vector --subset full --splits train,val
python -m passive_provenance.stages.fit_emulator --config configs/default.yaml --verifier-id d1_softmax --interface-id full_vector --subset full
python -m passive_provenance.stages.attack_blackbox --config configs/default.yaml --verifier-id d1_softmax --interface-id full_vector
```

Use each module's `--help` for its options. Where supported, `--subset smoke` selects a smaller setup check. Interface query caps apply separately to the training and validation splits.

For paired image-quality measurements, run `python -m passive_provenance.stages.perceptual_audit --help`. This stage takes a trained CLIP verifier checkpoint, clean operating-point metrics, and a triplet manifest. It reports LPIPS, SSIM, PSNR, perturbation norms, and attack success rates for all sources and initially correct sources.

## Acknowledgments

The implementation builds on PyTorch, OpenCLIP, Diffusers, SciPy, and LPIPS.

We thank the anonymous reviewers for their feedback, and the Edinburgh International Data Facility (EIDF) and the Data-Driven Innovation Programme at the University of Edinburgh for computing resources.

## Citation

```bibtex
@inproceedings{yao2026canpixels,
  title = {Can Pixels Alone Reveal Image Origin? Minimax Limits and Learnable Interfaces for Passive Provenance},
  author = {Yao, Kai},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2026},
  note = {Accepted},
  url = {https://arxiv.org/abs/2609.30997}
}
```
