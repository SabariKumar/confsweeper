"""Train the dihedral predictor (issue #20).

Usage:
    pixi run python scripts/train_dihedral_predictor.py \\
        --dataset data/processed/dihedral_predictor/dataset.pkl \\
        --out data/processed/dihedral_predictor/checkpoint.pt
"""

import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dihedral_predictor.train import train  # noqa: E402


@click.command()
@click.option("--dataset", default="data/processed/dihedral_predictor/dataset.pkl")
@click.option("--out", default="data/processed/dihedral_predictor/checkpoint.pt")
@click.option("--epochs", default=40)
@click.option("--batch_size", default=64)
@click.option("--lr", default=3e-4)
@click.option("--d_model", default=128)
@click.option("--n_layers", default=3)
@click.option("--window", default=1)
@click.option("--device", default="cuda")
def main(dataset, out, epochs, batch_size, lr, d_model, n_layers, window, device):
    train(
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
