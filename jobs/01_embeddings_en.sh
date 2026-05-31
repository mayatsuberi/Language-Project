#!/bin/bash
#SBATCH --job-name=embed_en
#SBATCH --mail-user=yahlizamero@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/sci/labs/arielgoldstein/yahlizamero/Language-Project/jobs/logs/01_en_%j.txt
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=32g
#SBATCH -c4
#SBATCH --time=0-2

set -e

# Redirect IPython/Jupyter away from the full home directory
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

source /usr/local/lmod/lmod/init/bash
module load cuda
module load nvidia

source /sci/labs/arielgoldstein/yahlizamero/venv/bin/activate
cd /sci/labs/arielgoldstein/yahlizamero/Language-Project

echo "=== Notebook 01: English Embeddings === $(date)"
jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=7200 \
    --output notebooks/01_Generate_Contextualized_Embeddings_EN_executed.ipynb \
    notebooks/01_Generate_Contextualized_Embeddings_EN.ipynb
echo "Done: $(date)"
