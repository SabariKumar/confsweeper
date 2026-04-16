import glob
import os
import random
import subprocess
import time
import uuid
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
import scipy as sp
from rdkit import Chem
from tqdm import tqdm

from confsweeper import *

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_SEED = 42
rd = random.Random()
rd.seed(RANDOM_SEED)


def make_confs(
    peptide_csv_path: os.PathLike | str,
    conf_save_dir: os.PathLike | str,
    n_confs: int = 1000,
    butina_thresh: float = 0.1,
):
    """
    Generates conformers and logs timing info.
    Params:
        peptide_db_path: os.PathLike|str : path to peptide defn csv
        conf_save_dir: os.PathLike|str : save directory for peptide conformations
    Returns:
        None
    """
    peptide_df = pd.read_csv(peptide_csv_path)
    if "uuids" not in peptide_df.columns:
        uuids = [
            str(uuid.UUID(int=rd.getrandbits(128), version=4).hex)
            for _ in peptide_df["SMILES"]
        ]
        peptide_df["uuid"] = uuids
    params = get_embed_params()
    hardware_opts = get_hardware_opts(
        preprocessingThreads=8,
        batch_size=1000,
        batchesPerGpu=2,
    )
    macemp = get_mace_calc()
    timings = []
    for smi, uuid in tqdm(zip(peptide_df["SMILES"], peptide_df["uuid"])):
        start_time = time.perf_counter()
        mol, conf_ids, pe = get_mol_PE(
            smi=smi,
            params=params,
            hardware_opts=hardware_opts,
            mace_calc=macemp,
            n_confs=n_confs,
            cutoff_dist=butina_thresh,
            gpu_clustering=True,
        )
        timings.append(time.perf_counter() - start_time)
