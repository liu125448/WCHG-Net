# 🌓 WCHG-Net（🔒 a limited-release version）

📄 PyTorch implementation for **WCHG-Net: Wavelet-Chebyshev Hypergraph Dual-Frequency Network for Visible-Infrared Person Re-identification**.

## 🔬 Method Names

- 🌊 `WAGP`: Wavelet-Aware Gated Purification.
- 📊 `CMGR`: Chebyshev Multi-band Graph Readout.
- 🆔 `Lid`: identity classification loss.
- 🎯 `Lqct`: quad-center triplet loss.
- 🎲 `Lher`: hyperedge entropy regularization.
- 📐 `Lhor`: hyperedge orthogonality regularization.

## 📁 Main Files

- 🚂 `train.py`: training entry.
- 🧪 `test.py`: evaluation entry.
- 🧠 `model_hgnn.py`: WCHG-Net backbone and training/inference forward pass.
- 🌊 `modules/WAGP.py`: WAGP module.
- 🕸️ `modules/HGNN.py`: CMGR, hypergraph construction, Chebyshev spectral readout, multi-granularity graph readout, and structural regularization.

## 🏋️ Training

```bash
python train.py --dataset sysu --gpu 0 --module full
python train.py --dataset regdb --gpu 0 --module full --trial 1
python train.py --dataset llcm --gpu 0 --module full

## 🏋️ test
python test.py --dataset sysu --mode all --resume sysu_module_full_wchgnet_p4_n6_lr_0.1_seed_0_SGD_1_best_fc_mAP.t --gpu 0
python test.py --dataset regdb --resume 'regdb_module_full_wchgnet_p4_n6_lr_0.1_seed_0_trial_{}_best_fc_mAP.t' --gpu 0
