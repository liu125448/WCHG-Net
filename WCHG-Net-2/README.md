# WCHG-Net-2

Public partial release for **WCHG-Net: Wavelet-Chebyshev Hypergraph Dual-Frequency Network for Visible-Infrared Person Re-identification**.

This repository is prepared for public sharing. It exposes the project layout,
training/evaluation entry points, data utilities, and method naming, while the
protected core implementations are intentionally redacted.

## Public Release Notice

WCHG-Net-2 is **not** a full reproduction release. The following parts are
deliberately replaced by interface-only placeholders:

- `modules/WAGP.py`: Wavelet-Aware Gated Purification (WAGP).
- `modules/HGNN.py`: CMGR, including hypergraph construction, Chebyshev
  multi-band spectral readout, multi-granularity graph readout, and structural
  regularization internals.
- `loss.py`: protected loss internals, including `Lqct` and related distance
  computations.
- trained weights, logs, and generated result files.

The placeholder modules raise `NotImplementedError` when executed. This is
intentional: the repository documents the public interface and code structure
without disclosing the complete method implementation.

## Method Names

- `WAGP`: Wavelet-Aware Gated Purification.
- `CMGR`: Chebyshev Multi-band Graph Readout.
- `Lid`: identity classification loss.
- `Lqct`: quad-center triplet loss.
- `Lher`: hyperedge entropy regularization.
- `Lhor`: hyperedge orthogonality regularization.

## Main Files

- `train.py`: training entry.
- `test.py`: evaluation entry.
- `model_hgnn.py`: WCHG-Net backbone and training/inference forward pass.
- `modules/WAGP.py`: WAGP public placeholder.
- `modules/HGNN.py`: CMGR/HGNN public placeholder.
- `loss.py`: public loss interfaces.

## Dataset Paths

Dataset paths are intentionally written as local placeholders. Put datasets
under `./dataset/` or update the paths in `train.py` and `test.py`.

Expected examples:

```text
dataset/SYSU-MM01/
dataset/RegDB/
dataset/LLCM/
```

## Example Commands

These commands show the expected interface of the private implementation. In
WCHG-Net-2 they will stop at the redacted modules.

```bash
python train.py --dataset sysu --gpu 0 --module full
python train.py --dataset regdb --gpu 0 --module full --trial 1
python train.py --dataset llcm --gpu 0 --module full
```

```bash
python test.py --dataset sysu --mode all --resume your_checkpoint.t --gpu 0
python test.py --dataset regdb --resume your_checkpoint.t --gpu 0
```

## Usage Terms

This public copy is provided for academic communication and project
identification only. Please do not copy, redistribute, or reconstruct the
redacted implementation without permission from the authors. If this project
or method naming helps your work, please cite the corresponding WCHG-Net paper
or contact the authors for the complete authorized version.
