"""Step 8 (issue #20): model-capacity / neighbour-window sweep.

Step 6 found prediction accuracy (not DG tolerance) is the limiter on landing the
dominant basin. This sweeps neighbour window and model capacity to push the val
metrics, especially peptide_all_ok (the end-to-end seeding proxy). Baseline
(window=1, d_model=128, n_layers=3, 40 ep) was peptide_all_ok=0.36.

Runs on CPU by default to avoid contending with a concurrent GPU MCMM run.
"""

import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dihedral_predictor.train import train  # noqa: E402

CONFIGS = [
    # label, window, d_model, n_layers
    ("w2_d128_l3", 2, 128, 3),
    ("w1_d256_l6", 1, 256, 6),
    ("w2_d256_l6", 2, 256, 6),
    ("w3_d256_l6", 3, 256, 6),
]


@click.command()
@click.option("--dataset", default="data/processed/dihedral_predictor/dataset.pkl")
@click.option("--epochs", default=50)
@click.option("--device", default="cpu")
def main(dataset, epochs, device):
    results = []
    for label, window, d_model, n_layers in CONFIGS:
        print(
            f"\n===== {label} (window={window} d_model={d_model} n_layers={n_layers}) =====",
            flush=True,
        )
        best = train(
            dataset,
            f"data/processed/dihedral_predictor/ckpt_{label}.pt",
            epochs=epochs,
            window=window,
            d_model=d_model,
            n_layers=n_layers,
            device=device,
        )
        results.append((label, best))
    print("\n===== SWEEP SUMMARY (baseline w1_d128_l3 = 0.36) =====", flush=True)
    for label, best in results:
        print(
            f"  {label:12s} peptide_all_ok={best['peptide_all_ok']:.3f} "
            f"phi_w1={best['phi_within1']:.2f} psi_w1={best['psi_within1']:.2f} "
            f"om={best['omega_acc']:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
