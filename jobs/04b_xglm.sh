#!/bin/bash
#SBATCH --job-name=xglm_embed
#SBATCH --mail-user=yahlizamero@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/sci/labs/arielgoldstein/yahlizamero/Language-Project/jobs/logs/04b_xglm_%j.txt
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=48g
#SBATCH -c4
#SBATCH --time=0-4

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

source /usr/local/lmod/lmod/init/bash
module load cuda
module load nvidia

source /sci/labs/arielgoldstein/yahlizamero/venv/bin/activate
cd /sci/labs/arielgoldstein/yahlizamero/Language-Project

echo "=== Notebook 04b: XGLM-1.7B === $(date)"
jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=18000 \
    --output notebooks/04b_Generate_XGLM_Embeddings_executed.ipynb \
    notebooks/04b_Generate_XGLM_Embeddings.ipynb
echo "Done: $(date)"
