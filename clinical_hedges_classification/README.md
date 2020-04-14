# COVID 19: BERT Based Classification Model for Vaccine / Treatment related Document Retrieval

**Note:** To understand this methodology, please read the methodology of our **document_nsp_text_sim model** before reading this one

The goal of this repo is to use novel language models pretrained on scientific text to effectively retrieve/filter from a bunch of Health Articles the ones which are most relevant to Treatment / Vaccination of Coronavirus (specifically COVID-19)

We model this problem as a text classification problem using the SciBERT model ([link](https://github.com/allenai/scibert)) which is a  type of BERT that is pre-trained on a huge corpus of scientific articles from semantic scholar. 

Classification in BERT uses the following format for input:\
`[CLS] + Text_A + [SEP]`

 The [Clinical Hedges dataset](https://hiru.mcmaster.ca/hiru/HIRU_Hedges_home.aspx) is a set of articles which are manually annotated for a bunch of categories like Format (Original, Review, etc), Purpose (Treatment, Etiology, etc), Rigor (How closely the methodology in the paper correspond to the purpose of the paper) and whether the papers are related to Human Health Care (HHC). 
 
 We use the Purpose column of this dataset (as described above) and label documents as Treatment and Not Treatment based on the annotations. Our model is then trained on this training dataset.
 
 Now, assuming our document-nsp-text-sim model (described previously) performs perfectly, we use the text documents categorized as Treatment or Vaccine by that model as the input to the Clinical Hedges model (here) at prediction time. Our Clinical Hedges model categorizes the text as Treatment or Not Treatment and if our dataset contains only Treatment and Vaccine related documents, our model will effectively categorize it as Treatment and Not Treatment (or Vaccine) related documents. Our architecture consists of the Fine Tuned SciBERT model with a FFN on top to give the probability scores for classification.
 
 The code in the doc should be run in the order specified to reproduce results. Moreover, the data and models are available in the dropbox folder mentioned in the respective folder's README. They can be downloaded from there to the specified location to run the code. 

`Note`: The code below had run on high performance GPU's and will take several minutes to execute.

`src/combine_nsp_results.py`: Code to combine all text which is classified as Vaccine or Therapeutics by the Query-Document NSP Model and prepare input to the Clinical Hedges classifier (run_classifier.py)

`src/run_classifier.py`: BERT finetuning runner for Text Classification
```
usage: run_classifier.py [-h] --data_dir DATA_DIR --model_dir MODEL_DIR
                         --vocab_dir VOCAB_DIR --task_name TASK_NAME
                         --output_dir OUTPUT_DIR --task_num TASK_NUM
                         [--cache_dir CACHE_DIR]
                         [--max_seq_length MAX_SEQ_LENGTH] [--do_train]
                         [--do_eval] [--do_lower_case]
                         [--train_batch_size TRAIN_BATCH_SIZE]
                         [--eval_batch_size EVAL_BATCH_SIZE]
                         [--learning_rate LEARNING_RATE]
                         [--num_train_epochs NUM_TRAIN_EPOCHS]
                         [--warmup_proportion WARMUP_PROPORTION] [--no_cuda]
                         [--local_rank LOCAL_RANK] [--seed SEED]
                         [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                         [--fp16] [--loss_scale LOSS_SCALE]
                         [--server_ip SERVER_IP] [--server_port SERVER_PORT]
 
```

`Run_BERT_CH.sh`: Bash Script to execute run_classifier.py using GPU's. \
```usage: bash Run_BERT_CH.sh ``` or ```sbatch Run_BERT_CH.sh (for SLURM Support)``` 

`src/combine_results_covid.py`: Code that reads the predicted output from the Clinical Hedges model and writes it along with the Text as output

`../document_nsp_text_sim/src/convert_csv_results_json.py`: Code to convert the CSV formatted results to JSON for faster retrieval during Ensemble
