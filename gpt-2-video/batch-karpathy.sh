#!/bin/bash
#SBATCH --qos turing
#SBATCH --account vjgo8416-karpathy
#SBATCH --time 0:15:0
#SBATCH --nodes 1
#SBATCH --gpus 4
#SBATCH --cpus-per-gpu 9
#SBATCH --mem 16384
#SBATCH --job-name karpathy-watching
#SBATCH --output karpathy-%j.out

# Execute using:
# sbatch batch-karpathy.sh

module purge
module load baskerville
module load bask-apps/live
module load PyTorch/2.1.2-foss-2022b-CUDA-11.8.0

pushd /bask/projects/v/vjgo8416-karpathy/${USER}/minGPT/gpt-2-video

python3 -m venv venv
source ./venv/bin/activate
pip install pip --upgrade
pip install -r requirements.txt
export OMP_NUM_THREADS=${SLURM_CPUS_PER_GPU}

echo
echo "######################################"
echo "Starting"
echo "######################################"
echo

# Track GPU metrics
stdbuf -o0 nvidia-smi dmon -o TD -s puct -d 1 > gpu-${SLURM_JOB_ID}.txt &

# Track CPU metrics
stdbuf -o0 vmstat -t 1 -y > cpu-${SLURM_JOB_ID}.txt &

# Execute the training
python3 \
    -m torch.distributed.launch \
    --standalone \
    --nproc_per_node=${SLURM_GPUS} \
    train_gpt2.py

echo
echo "######################################"
echo "Done"
echo "######################################"
echo

deactivate
popd

