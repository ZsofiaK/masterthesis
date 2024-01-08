!/bin/bash
SBATCH --gpus=1
SBATCH --partition=gpu
SBATCH --time=14:00:00
SBATCH -o exercise_%j.out

echo "gpus $SLURM_GPUS on node: $SLURM_GPUS_ON_NODE"
echo "nodes nnodes: $SLURM_NNODES, nodeid: $SLURM_NODEID, nodelist $SLURM_NODELIST"
echo "cpus on node: $SLURM_CPUS_ON_NODE per gpu $SLURM_CPUS_PER_GPU per task $SLURM_CPUS_PER_TASK omp num thread $OMP_NUM_THREADS"
echo "tasks per node $SLURM_TASKS_PER_NODE pid $SLURM_TASK_PID"

# activate your environment
source $HOME/.bashrc
conda activate test # your conda venv

echo "start running exercise"
python /Users/katonazsofia/Desktop/Master's Thesis/masterthesis/Programming exercise/Exercise.py
echo "end running exercise"