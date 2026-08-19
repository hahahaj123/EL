# EL Diffusion Distillation

This is the runnable EL research repository. It contains the original EL training loops, the upstream SiD/SiDA/EDM2 implementations required by those loops, and the Diff-Instruct baseline. Every method has an explicit command-line entry point; no training code is left as an unreferenced script.

## Repository layout

| Method | Entry point | Launcher |
| --- | --- | --- |
| Diff-Instruct baseline | `di_train.py` | command below |
| EL custom Diff-Instruct loop | `di_train_custom.py` | command below |
| SiD | `sid_train.py` | `run_sid.sh` |
| SiDA | `sida_train.py` | `run_sida.sh` |
| EL custom SiDA loop | `sida_train_custom.py` | command below |
| EL custom conditional SiDA loop | `sida_train_custom_cond.py` | command below |
| SiD-SiDA | `sida_train.py` | `run_sid_sida.sh` |
| EDM2 SiDA | `sida_edm2_train.py` | `run_sida_edm2.sh` |

The custom loops are in `experiments/`. Shared runtime code is in `dnnlib/`, `torch_utils/`, `training/`, `sida_training/`, `metrics/`, and `metrics_edm2/`.

## Environment

Use the pinned environment used by the upstream SiD branch:

```bash
conda env create -f environment.yaml
conda activate sida
```

Training requires a CUDA GPU. Use `python --version`, `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`, and the CLI checks below before starting a long run.

## Data and checkpoints

Datasets and model weights are not committed. Convert datasets to the EDM ZIP format and place them at the paths in `data/datasets/`, or pass an explicit `--data` path. Download teacher/SiD checkpoints from the URLs in the launchers or pass local paths with `--edm_model`, `--resume`, and `--sid_model`. Outputs are written below `data/image_experiment/` and are ignored by Git.

The required preparation details and published checkpoints are documented by the upstream projects:

- SiD/SiDA: https://github.com/mingyuanzhou/SiD/tree/sida
- Diff-Instruct: https://github.com/pkulwj1994/diff_instruct
- EDM datasets: https://github.com/NVlabs/edm

## Reproduction commands

Run commands from the repository root. Adjust `CUDA_VISIBLE_DEVICES`, `--nproc_per_node`, batch sizes, and paths to match the machine.

### SiD-SiDA preset

```bash
bash run_sid_sida.sh cifar10-uncond
```

### EL custom SiDA loop

The custom entry point accepts the same arguments as `sida_train.py` and uses `experiments/sida_training_loop_custom.py`:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  sida_train_custom.py \
  --data ./data/datasets/cifar10-32x32.zip \
  --outdir ./data/image_experiment/custom-sida/cifar10-uncond \
  --edm_model cifar10-uncond \
  --batch 256 --batch-gpu 32 --duration 100 \
  --metrics fid50k_full
```

For class-conditional experiments use `sida_train_custom_cond.py --cond 1` and the corresponding labeled dataset.

### Diff-Instruct baseline and EL custom loop

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  di_train.py --outdir ./data/image_experiment/diff-instruct/cifar10 \
  --data ./data/datasets/cifar10-32x32.zip --arch ddpmpp \
  --batch 128 --edm_model cifar10-uncond --cond 0 \
  --metrics fid50k_full --init_sigma 1.0 --fp16 0

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  di_train_custom.py --outdir ./data/image_experiment/el-diff-instruct/cifar10 \
  --data ./data/datasets/cifar10-32x32.zip --arch ddpmpp \
  --batch 128 --edm_model cifar10-uncond --cond 0 \
  --metrics fid50k_full --init_sigma 1.0 --fp16 0
```

## Verification

Before training, verify all entry points import correctly:

```bash
python -m compileall -q .
python sida_train.py --help
python sida_train_custom.py --help
python sida_train_custom_cond.py --help
python di_train.py --help
python di_train_custom.py --help
```

These checks validate the repository and CLI wiring; they do not replace a GPU training run. For strict reproduction, record the exact Git commit, environment YAML, dataset archive checksum, checkpoint URL or checksum, GPU count, and launcher arguments in the output directory.

## License and provenance

See `LICENSE.txt` and the headers in individual upstream files. The shared framework is derived from the `sida` branch of SiD and Diff-Instruct; the EL-specific loops are kept under `experiments/` with their original notices.
