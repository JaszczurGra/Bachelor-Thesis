JOB_ID=$(sbatch --parsable slurm_generate_folders.sh "https://wandb.ai/j-boro-poznan-university-of-technology/Bachelor-Thesis-src_model/sweeps/xrqywefs/table?nw=nwuserjaszczurgra" 512 "")
echo "Submitted sweep vis: $JOB_ID"
JOB_ID=$(sbatch --parsable slurm_generate_folders.sh "https://wandb.ai/j-boro-poznan-university-of-technology/Bachelor-Thesis-src_model/sweeps/v7xcqp8a/table?nw=nwuserjaszczurgra" 16 "")
echo "Submitted sweep vis: $JOB_ID"
JOB_ID=$(sbatch --parsable slurm_generate_folders.sh "https://wandb.ai/j-boro-poznan-university-of-technology/Bachelor-Thesis-src_model/sweeps/xrqywefs/table?nw=nwuserjaszczurgra" 512 "data/grouchy_penguin_23-01-2026_00:43:35")
echo "Submitted sweep vis: $JOB_ID"
JOB_ID=$(sbatch --parsable slurm_generate_folders.sh "https://wandb.ai/j-boro-poznan-university-of-technology/Bachelor-Thesis-src_model/sweeps/v7xcqp8a/table?nw=nwuserjaszczurgra" 16 "data/grouchy_penguin_23-01-2026_00:43:35")
echo "Submitted sweep vis: $JOB_ID"
