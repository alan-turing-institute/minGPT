#!/bin/bash
#SBATCH --qos turing
#SBATCH --account vjgo8416-karpathy
#SBATCH --time 0:15:0
#SBATCH --nodes 1
#SBATCH --gpus 2
#SBATCH --mem 16384
#SBATCH --job-name karpathy-watching

# Execute using:
# sbatch -o karpathy.out batch-karpathy.sh

module purge
module load baskerville
module load bask-apps/live
module load PyTorch/2.1.2-foss-2022b-CUDA-11.8.0

pushd /bask/projects/v/vjgo8416-karpathy/${USER}/minGPT/gpt-2-video

python3 -m venv venv
source ./venv/bin/activate
pip install pip --upgrade
pip install -r requirements.txt

echo
echo "######################################"
echo "Starting"
echo "######################################"
echo

# Track GPU metrics
stdbuf -o0 nvidia-smi dmon -o TD -s puct -d 1 > dmon.txt &

# Track CPU metrics
stdbuf -o0 vmstat -t 1 -y > cpu.txt &

echo "GPUs: ${SLURM_GPUS}"

torchrun --standalone --nproc_per_node=${SLURM_GPUS} train_gpt2.py

echo
echo "######################################"
echo "Done"
echo "######################################"
echo

deactivate
popd

