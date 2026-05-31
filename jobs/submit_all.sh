#!/bin/bash
# Run from moriah AFTER transferring files and setting up the venv:
#   ssh yahlizamero@moriah-gw.cs.huji.ac.il
#   cd /sci/labs/arielgoldstein/yahlizamero/Language-Project
#   bash jobs/submit_all.sh

set -e
cd /sci/labs/arielgoldstein/yahlizamero/Language-Project
mkdir -p jobs/logs

echo "Submitting pipeline on Moriah..."

JID_01=$(sbatch --parsable jobs/01_embeddings_en.sh)
JID_02=$(sbatch --parsable jobs/02_embeddings_he.sh)
JID_03=$(sbatch --parsable jobs/03_embeddings_ar.sh)
JID_04=$(sbatch --parsable jobs/04_sliding_window.sh)
JID_04b=$(sbatch --parsable jobs/04b_xglm.sh)

echo "  [01] EN embeddings   → job $JID_01"
echo "  [02] HE embeddings   → job $JID_02"
echo "  [03] AR embeddings   → job $JID_03"
echo "  [04] Sliding window  → job $JID_04"
echo "  [04b] XGLM           → job $JID_04b"

DEPS_05="${JID_01}:${JID_02}:${JID_03}:${JID_04}:${JID_04b}"
JID_05=$(sbatch --parsable --dependency=afterok:${DEPS_05} jobs/05_residuals.sh)
echo "  [05] Residuals       → job $JID_05  (after $DEPS_05)"

JID_06=$(sbatch --parsable --dependency=afterok:${JID_05} jobs/06_encoding.sh)
echo "  [06] Encoding        → job $JID_06  (after $JID_05)"

JID_07=$(sbatch --parsable --dependency=afterok:${JID_06} jobs/07_plots.sh)
echo "  [07] Plots           → job $JID_07  (after $JID_06)"

echo ""
echo "Monitor:  sacct -u \$USER"
echo "Cancel:   scancel <JOB_ID>"
echo "Watch:    tail -f jobs/logs/06_encoding_${JID_06}.txt"
