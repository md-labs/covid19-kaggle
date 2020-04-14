#!/bin/bash
#SBATCH -N 1
###SBATCH -n 3
###SBATCH -p cidsegpu1
###SBATCH -p physicsgpu1
###SBATCH -p asinghargpu1
###SBATCH -p wzhengpu1
#SBATCH -p cidsegpu2
#SBATCH -q wildfire

#SBATCH --gres=gpu:2

#SBATCH -J BERT_Agave_COVID
#SBATCH -o BERT_Agave.OUT
#SBATCH -e BERT_Agave.ERROR

#SBATCH -t 1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aambalav@asu.edu

module load anaconda3/5.3.0

pip install --user pytorch-pretrained-bert==0.6.0

python src/run_nsp.py --model_file=models/scibert_scivocab_uncased --bert_model=models/scibert_scivocab_uncased --do_lower_case --task_name=covid --data_dir=data/COVID_NSP_Data --learning_rate=2e-5 --num_train_epochs=10 --output_dir=models/scibert_scivocab_uncased/ --cache_dir=./BERT_CACHE --eval_batch_size=16 --max_seq_length=512 --train_batch_size=16 --do_eval
