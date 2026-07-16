# From-Scratch CNN for 16x16 EMNIST Characters

This project implements a small CNN from scratch with NumPy for 47-class handwritten character recognition on binary 16x16 images.

## Main Commands

```bash
python train.py
python evaluate.py
python run_all_evaluations.py
python test_draw.py
```

## Project Layout

- `train.py`, `evaluate.py`, `run_experiments.py`: training and evaluation entry points.
- `cnn_models.py`, `layers.py`: from-scratch CNN model and layers.
- `test_draw.py`, `test_brain.py`: interactive drawing demo and user-feedback collection.
- `data/`: training dataset and user-feedback samples.
- `models/`: trained model weights.
- `artifacts/`: packaged Kaggle/evaluation ZIP outputs.
- `report_assets/`: figures used by `report.tex`.
- `docs/`: planning notes and supporting documentation.

Generated outputs are written to `results/` and are ignored by Git. The downloadable evaluation package is written to `artifacts/cnn_evaluation_results.zip`.
