# EL: reproducible diffusion distillation experiments

This repository contains the experiment code from `E:\aaa论文\代码`, together with the runtime files required by the SiD/SiDA and Diff-Instruct training entry points. The upstream SiD documentation is kept in `README.md`; this file documents the EL-specific layout and reproduction workflow.

## Contents

- `sida_train.py`, `run_sid_sida.sh`: SiD-SiDA training entry point and dataset presets.
- `sid_train.py`, `run_sid.sh`, `run_sida.sh`: upstream SiD/SiDA entry points.
- `di_train.py`: Diff-Instruct training entry point.
- `training/di_training_loop_DM.py`: the supplied custom Diff-Instruct training loop.
- `sida_training_loop_DM.py` and `sida_training_loop_DM_con.py`: supplied custom SiDA loops (the original filenames are also retained for traceability).
- `dnnlib/`, `torch_utils/`, `training/`, `sida_training/`, `metrics/`: required runtime modules.
- `notebooks/run_experiment.ipynb`: the supplied generation command notebook.

## Environment

The tested dependency specification is `environment.yaml` (Python 3.12, PyTorch 2.4.1, CUDA packages supplied by the PyTorch channel). Create it with:

```bash
conda env create -f environment.yaml
conda activate sida
```

For a CPU-only smoke test, install the matching PyTorch build separately; full training requires a CUDA-capable GPU and multiple GPUs for the preset scripts.

## Data and checkpoints

Training data is intentionally not committed. Prepare the EDM-format ZIP archives described in `README.md` and place them under `data/datasets/` using the names expected by `run_sid_sida.sh`. Pretrained teacher and SiD checkpoints are downloaded from the URLs in the scripts or supplied through `--sid_model`. Generated checkpoints and experiment logs should remain outside Git.

## Reproduce

From the repository root, after activating the environment and preparing data:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 sida_train.py \
  --data ./data/datasets/cifar10-32x32.zip \
  --outdir ./data/image_experiment/sida-train-runs/cifar10-uncond
```

The complete parameter presets are in `run_sid_sida.sh`; run one with `bash run_sid_sida.sh cifar10-uncond` after adjusting GPU count and paths for the local machine. The Diff-Instruct baseline can be launched with the command in `README.md`, using `di_train.py`.

## Provenance

The runtime scaffold is based on the `sida` branch of [mingyuanzhou/SiD](https://github.com/mingyuanzhou/SiD/tree/sida), and the Diff-Instruct modules are from [pkulwj1994/diff_instruct](https://github.com/pkulwj1994/diff_instruct). Checkpoint URLs and dataset preparation instructions remain those projects' published resources. No datasets, pretrained weights, or generated outputs are included.

## License

See `LICENSE.txt` for the upstream SiD license and the corresponding upstream files for their individual notices. The custom files retain their original headers where present.
