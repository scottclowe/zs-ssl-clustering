#!/bin/bash
#SBATCH --partition=cpu             # Which node partitions to use.
#SBATCH --nodes=1                   # Number of nodes to request.
#SBATCH --tasks-per-node=1          # Number of processes to spawn per node.
#SBATCH --cpus-per-task=2           # Number of CPUs per task.
#SBATCH --mem=2G                    # Total RAM.
#SBATCH --time=96:00:00             # You must specify a maximum run-time if you want to run for more than 2h
#SBATCH --output=slogs/%x__%A_%a.out
                                    # %x=job-name, %A=job ID, %a=array value, %n=node rank, %t=task rank, %N=hostname
                                    # Note: You must create output directory "slogs" before launching job, otherwise it will immediately
                                    # fail without an error message.
                                    # Note: If you specify --output and not --error, then both STDOUT and STDERR will both be sent to the
                                    # file specified by --output.
#SBATCH --array=0                   # Use array to run multiple jobs that are identical except for $SLURM_ARRAY_TASK_ID.
                                    # We use this to set the seed. You can run multiple seeds with --array=0-4, for example.
#SBATCH --open-mode=append          # Use append mode otherwise preemption resets the checkpoint file.
#SBATCH --job-name=zs-ssl-clus      # Set this to be a shorthand for your project's name.

# Manually define the project name.
# This must also be the name of your conda environment used for this project.
PROJECT_NAME="zs-ssl-clustering"
# Automatically convert hyphens to underscores, to get the name of the project directory.
PROJECT_DIRN="${PROJECT_NAME//-/_}"

# Exit the script if any command hits an error
set -e

# sbatch script for Vector
# Based on
# https://github.com/VectorInstitute/TechAndEngineering/blob/master/benchmarks/resnet_torch/sample_script/script.sh
# https://github.com/VectorInstitute/TechAndEngineering/blob/master/checkpoint_examples/PyTorch/launch_job.slrm
# https://github.com/VectorInstitute/TechAndEngineering/blob/master/checkpoint_examples/PyTorch/run_train.sh

# Store the time at which the script was launched, so we can measure how long has elapsed.
start_time="$SECONDS"

echo "Job $SLURM_JOB_NAME ($SLURM_JOB_ID) begins on $(hostname), submitted from $SLURM_SUBMIT_HOST ($SLURM_CLUSTER_NAME)"

# Print slurm config report (SLURM environment variables, some of which we use later in the script)
# By sourcing the script, we execute it as if its code were here in the script
# N.B. This script only prints things out, it doesn't assign any environment variables.
echo "Running slurm/utils/report_slurm_config.sh"
source "slurm/utils/report_slurm_config.sh"
# Print repo status report (current branch, commit ref, where any uncommitted changes are located)
# N.B. This script only prints things out, it doesn't assign any environment variables.
echo "Running slurm/utils/report_repo.sh"
source "slurm/utils/report_repo.sh"
echo ""
if false; then
    # Print disk usage report, to catch errors due to lack of file space.
    # This is disabled by default to prevent confusing new users with too
    # much output.
    echo "------------------------------------"
    echo "df -h:"
    df -h --output=target,pcent,size,used,avail,source | head -n 1
    df -h --output=target,pcent,size,used,avail,source | tail -n +2 | sort -h
    echo ""
fi
echo "-------- Input handling ------------------------------------------------"
date
echo ""
# Use the SLURM job array to select the seed for the experiment
SEED="$SLURM_ARRAY_TASK_ID"
if [[ "$SEED" == "" ]];
then
    SEED=0
fi
echo "SEED = $SEED"

# Any arguments provided to sbatch after the name of the slurm script will be
# passed through to the main script later.
# (The pass-through works like *args or **kwargs in python.)
echo "Pass-through args: ${@}"
echo ""
echo "-------- Activating environment ----------------------------------------"
date
echo ""
echo "Running ~/.bashrc"
source ~/.bashrc
echo ""
# Activate virtual environment
ENVNAME="$PROJECT_NAME"
echo "Activating conda environment $ENVNAME"
conda activate "$ENVNAME"
echo ""
# Print env status (which packages you have installed - useful for diagnostics)
# N.B. This script only prints things out, it doesn't assign any environment variables.
echo "Running slurm/utils/report_env_config.sh"
source "slurm/utils/report_env_config.sh"

# Set the JOB_LABEL environment variable
# N.B. This script assigns the $JOB_LABEL environment variables that we use later.
echo "Running slurm/utils/set_job-label.sh"
source "slurm/utils/set_job-label.sh"

echo ""
echo "-------- Begin main script ---------------------------------------------"
date

python "$PROJECT_DIRN/cluster.py" \
    --seed="$SEED" \
    --log-wandb \
    --run-name="$SLURM_JOB_NAME" \
    --run-id="$JOB_ID" \
    "${@}"

echo ""
echo "------------------------------------------------------------------------"
echo ""
echo "Job $SLURM_JOB_NAME ($SLURM_JOB_ID) finished, submitted from $SLURM_SUBMIT_HOST ($SLURM_CLUSTER_NAME)"
date
echo "------------------------------------"
elapsed=$(( SECONDS - start_time ))
eval "echo Total elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"
echo "========================================================================"
