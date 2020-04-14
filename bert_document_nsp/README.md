# BERT_Document_NSP

The goal of this repo is to use novel language models pretrained on scientific text to effectively retrieve/filter from a bunch of Health Articles the ones which are most relevant to Treatment / Vaccination of Coronavirus (specifically COVID-19)

The code in the doc should be run in the order specified to reproduce results. Moreover, the data and models are available in the dropbox folder mentioned in the respective folder's README. They can be downloaded from there to the specified location to run the code. 

`Note`: The code below had run on high performance GPU's and take several minutes to execute.

`src/create_nsp_data.py`: Code to create the input to the BERT Next Sentence Prediction Model (run_nsp.py)

`src/run_nsp.py`: BERT finetuning runner for Next Sentence Prediction
```
usage: run_nsp.py [-h] --model_file MODEL_FILE --data_dir DATA_DIR
                  --bert_model BERT_MODEL --task_name TASK_NAME --output_dir
                  OUTPUT_DIR [--cache_dir CACHE_DIR]
                  [--max_seq_length MAX_SEQ_LENGTH] [--do_train] [--do_eval]
                  [--do_lower_case] [--train_batch_size TRAIN_BATCH_SIZE]
                  [--eval_batch_size EVAL_BATCH_SIZE]
                  [--learning_rate LEARNING_RATE]
                  [--num_train_epochs NUM_TRAIN_EPOCHS]
                  [--warmup_proportion WARMUP_PROPORTION] [--no_cuda]
                  [--local_rank LOCAL_RANK] [--seed SEED]
                  [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                  [--fp16] [--loss_scale LOSS_SCALE] [--server_ip SERVER_IP]
                  [--server_port SERVER_PORT]

Note: bert_model is mainly used for getting the vocab file. model_file is the actual input model used 
```

`Run_BERT_COVID_NSP.sh`: Bash Script to execute run_nsp.py using GPU's. \
```usage: bash Run_BERT_COVID_NSP.sh ``` or ```sbatch Run_BERT_COVID_NSP.sh``` 

`src/analyze_nsp_results.py`: Code to read the output csv from run_nsp.py and metadata.csv. It filters papers based on a threshold set as 0.999 for
either Treatment or Vaccine Queries. It assigns label based on whichever is higher beyond 0.999 and O if it isn't beyond
the threshold.

`src/convert_csv_results_json.py`: Code to convert the CSV formatted results to JSON for faster retrieval during Ensemble

`src/lm_finetuning/`: Contains code to do MLM/NSP Intermediate FineTuning
