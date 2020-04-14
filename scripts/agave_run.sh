#!/bin/bash
#SBATCH -N 1
#SBATCH -n 24
#SBATCH -p cidsegpu3
###SBATCH -p physicsgpu1
#SBATCH -q wildfire

#SBATCH --gres=gpu:4

#SBATCH -J BERT_Agave
#SBATCH -o MTDNN_Agave.OUT
#SBATCH -e MTDNN_Agave.ERROR

#SBATCH -t 0-01:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mihirparmar@asu.edu

export OMP_NUM_THREADS=24
module load anaconda3/5.3.0

#pip3 install --user tensorflow==2.0.0
#pip3 install --user tensorflow-gpu==2.0.0
#pip3 install --user transformers
#pip3 install --user docopt==0.6.1
#pip3 install --user astropy==4.0
#pip3 install --user tensorboard==2.0.0
#pip3 install --user tensorflow-estimator==2.0.0

python3 bert_covid.py --data_dir=./data --bert_model=bert-base-uncased --output_dir=./output_128 --max_seq_length=128 --num_train_epochs=10 --do_eval --train_batch_size=8 --eval_batch_size=1
