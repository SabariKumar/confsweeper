"""Train the SEPARATE side-chain chi predictor (issue #20, Step 8).

Independent of the backbone DihedralPredictor (no shared weights), so backbone
fidelity is unaffected. Reads chi targets from the same dataset pickle.

Usage:
    pixi run python scripts/train_chi_predictor.py \\
        --dataset data/processed/dihedral_predictor/dataset.pkl \\
        --out data/processed/dihedral_predictor/chi_checkpoint.pt
"""

import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dihedral_predictor.train import train_chi  # noqa: E402


@click.command()
@click.option("--dataset", default="data/processed/dihedral_predictor/dataset.pkl")
@click.option("--out", default="data/processed/dihedral_predictor/chi_checkpoint.pt")
@click.option("--epochs", default=50)
@click.option("--batch_size", default=64)
@click.option("--lr", default=3e-4)
@click.option("--d_model", default=256)
@click.option("--n_layers", default=6)
@click.option("--window", default=2)
@click.option("--device", default="cuda")
def main(dataset, out, epochs, batch_size, lr, d_model, n_layers, window, device):
    train_chi(
        dataset,
        out,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        d_model=d_model,
        n_layers=n_layers,
        window=window,
        device=device,
    )


if __name__ == "__main__":
    main()
