#!/bin/bash
# filepath: /home/jaszczurgra/Documents/Programing/Bachelor-Thesis/launch_sweep_waves.sh

eval "$(/mnt/storage_6/project_data/pl0467-01/soft/miniconda3/bin/conda shell.bash hook)"
conda activate planning_diffusion


#could have 2 diffrent configs for the 2 waves f.e more epocohs for fine tuning in second wave 

# Create sweep once
SWEEP_OUTPUT=$(wandb sweep sweep_config.yaml 2>&1)
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oP 'wandb agent \K.*')

echo "Sweep ID: $SWEEP_ID"

# Wave 1: Fast exploration (10 agents, 3 hours)
echo "Launching Wave 1: Fast exploration"
sbatch --array=0-2 --time=0-15:00:00 <<EOF
#!/bin/bash
#SBATCH --job-name=diffusion_planning_sweep
#SBATCH --output=log/diffusion_planning_sweep_%A_%a.out
#SBATCH --error=log/train_%j.err
#SBATCH --partition=tesla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=0-15:00:00

eval "\$(/mnt/storage_6/project_data/pl0467-01/soft/miniconda3/bin/conda shell.bash hook)"
conda activate planning_diffusion
wandb agent $SWEEP_ID
EOF

# # Wait for wave 1 to finish
# sleep 3h

# # Wave 2: Refinement (5 agents, 6 hours)
# echo "Launching Wave 2: Refinement"
# sbatch --array=0-4 --time=0-06:00:00 <<EOF
# #!/bin/bash
# #SBATCH --job-name=sweep_wave2
# #SBATCH --output=log/wave2_%A_%a.out
# #SBATCH --partition=gpu
# #SBATCH --gres=gpu:1
# #SBATCH --mem=32G

# eval "\$(/mnt/storage_6/project_data/pl0467-01/soft/miniconda3/bin/conda shell.bash hook)"
# conda activate planning_diffusion
# wandb agent $SWEEP_ID
# EOF

# echo "Launched 2-wave sweep"
# echo "View at: https://wandb.ai"