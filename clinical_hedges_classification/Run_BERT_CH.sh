#!/bin/bash
#SBATCH -N 1
#SBATCH -n 3
#SBATCH -p cidsegpu2
###SBATCH -p physicsgpu1
#SBATCH -q wildfire

#SBATCH --gres=gpu:2

#SBATCH -J BERT_Agave_CH_ITL_Task_3
#SBATCH -o BERT_Agave_ITL_Task_3.OUT
#SBATCH -e BERT_Agave_ITL_Task_3.ERROR

#SBATCH -t 1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your-email-id@asu.edu

export OMP_NUM_THREADS=3
module load anaconda3/5.3.0

pip install --user pytorch-pretrained-bert==0.6.0

python src/run_classifier.py  --task_num=3 --model_file=models/scibert_scivocab_uncased --vocab_dir=models/scibert_scivocab_uncased --do_lower_case --task_name=clinicalhedges --data_dir=data/Pred_Data_COVID --learning_rate=2e-5 --num_train_epochs=10 --output_dir=models/SciBERT_Trained_Treatment_Model/ --eval_batch_size=16 --max_seq_length=400 --train_batch_size=16 --do_eval
