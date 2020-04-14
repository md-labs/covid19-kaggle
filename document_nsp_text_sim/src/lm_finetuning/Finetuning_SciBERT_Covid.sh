#!/bin/bash
#SBATCH -N 1
#SBATCH -n 3
#SBATCH -p physicsgpu2
###SBATCH -p physicsgpu1
###SBATCH -p rcgpu3
#SBATCH -q wildfire

#SBATCH --gres=gpu:4

#SBATCH -J COVID_FineTuning
#SBATCH -o COVID_FineTuning.OUT
#SBATCH -e COVID_FineTuning.ERROR

#SBATCH -t 1-0:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aambalav@asu.edu

export OMP_NUM_THREADS=3
module load anaconda3/5.3.0
pip install --user pytorch-pretrained-bert==0.6.0

###python3 pregenerate_training_data.py --train_corpus FineTune.txt --bert_model ../../trained_model/scibert_scivocab_uncased --do_lower_case --output_dir training/ --epochs_to_generate 10 --max_seq_len 400

python3 finetune_on_pregenerated.py --pregenerated_data training/ --bert_model ../../trained_model/scibert_scivocab_uncased --do_lower_case --output_dir finetuned_scibert/ --epochs 10
