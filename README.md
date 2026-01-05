GPU accelerated conformer clustering + energy prediction with nvmolkit and MACE

This project is set up to use pixi for dependency management. To install, (after installing pixi):
- `git clone` the repo
- `cd` to the repo root
- `pixi install`
- If running on an HPC system's CPU nodes/systems without a graphics card, you may need to mock the CUDA drivers: `export CONDA_OVERRIDE_CUDA=12.2`
