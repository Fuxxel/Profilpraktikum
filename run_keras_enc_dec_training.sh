#!/bin/bash

#SBATCH --time=24:00:00   # walltime
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks=1      # limit to one node
#SBATCH --cpus-per-task=4  # number of processor cores (i.e. threads)
#SBATCH --partition=ml
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2500M   # memory per CPU core
#SBATCH -A p_da_getml
#SBATCH --mail-user=marvin.arnold@mailbox.tu-dresden.de   # email address
#SBATCH --mail-type=BEGIN,END,FAIL

module load modenv/ml
module load TensorFlow

python main_keras.py -a enc_dec -s trained_models/Lager4 -i Data/Lager4/training -b 256 -e 700