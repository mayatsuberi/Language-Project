#!/bin/bash
#SBATCH --job-name=residuals
#SBATCH --mail-user=yahlizamero@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/sci/labs/arielgoldstein/yahlizamero/Language-Project/jobs/logs/05_residuals_%j.txt
#SBATCH --partition=glacier
#SBATCH --mem=32g
#SBATCH -c8
#SBATCH --time=0-1

set -e

export IPYTHONDIR=/sci/labs/arielgoldstein/yahlizamero/.ipython
export JUPYTER_CONFIG_DIR=/sci/labs/arielgoldstein/yahlizamero/.jupyter
export JUPYTER_DATA_DIR=/sci/labs/arielgoldstein/yahlizamero/.local/share/jupyter
export TMPDIR=/sci/labs/arielgoldstein/yahlizamero/tmp
export HF_HOME=/sci/labs/arielgoldstein/yahlizamero/.cache/huggingface
export XDG_CACHE_HOME=/sci/labs/arielgoldstein/yahlizamero/.cache
export XDG_CONFIG_HOME=/sci/labs/arielgoldstein/yahlizamero/.config
export MPLCONFIGDIR=/sci/labs/arielgoldstein/yahlizamero/.config/matplotlib
export TORCH_HOME=/sci/labs/arielgoldstein/yahlizamero/.cache/torch
export TRITON_CACHE_DIR=/sci/labs/arielgoldstein/yahlizamero/.cache/triton
export JUPYTER_RUNTIME_DIR=/sci/labs/arielgoldstein/yahlizamero/tmp/jupyter_runtime_$$
mkdir -p "$JUPYTER_RUNTIME_DIR"

source /sci/labs/arielgoldstein/yahlizamero/venv/bin/activate
cd /sci/labs/arielgoldstein/yahlizamero/Language-Project

echo "=== Notebook 05: Residuals === $(date)"
jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=3600 \
    --output notebooks/05_Projection_Residuals_Contextual_executed.ipynb \
    notebooks/05_Projection_Residuals_Contextual.ipynb
echo "Done: $(date)"
