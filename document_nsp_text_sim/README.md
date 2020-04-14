# COVID 19: BERT Based Query-Document Text Similarity using NSP for Treatment/Vaccine related Document Retrieval

The goal of this repo is to use novel language models pretrained on scientific text to effectively retrieve/filter from a bunch of Health Articles the ones which are most relevant to Treatment / Vaccination of Coronavirus (specifically COVID-19)

We model this problem using the Next Sentence Prediction (NSP) property that is used as a pretraining strategy in the novel transformer Language model- BERT ([paper](https://arxiv.org/abs/1810.04805)). We particularly use SciBERT ([link](https://github.com/allenai/scibert)) which is a  type of BERT that is pre-trained on a huge corpus of scientific articles from semantic scholar. 

NSP in BERT is pre-trained using the following format for input:\
`[CLS] + Text_A + [SEP] + Text_B + [SEP]`

We use hand-picked sentences from BioMedical Papers from renowned conferences and use these as the query (Text_A). These queries are formulated such that they signify strongly Therapeutics and Vaccines (one for each). We then use the title + abstract + journal_id of the papers in the COVID dataset ([link](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)) as Text_B.
 
 This combined text is input along with the query of Vaccine and Therapeutics separately and a Feed Forward Network acts as the NSP head which with the help of a softmax layer gives similarity probability scores between the two text elements. A threshold of 0.999 is set and if the similarity score exceed this, we classify it as relevant to Treatment or Vaccine according to which query it is most similar to.  

The code in the doc should be run in the order specified to reproduce results. Moreover, the data and models are available in the dropbox folder mentioned in the respective folder's README. They can be downloaded from there to the specified location to run the code. 

`Note`: The code below had run on high performance GPU's and will take several minutes to execute.

`src/create_nsp_data.py`: Code to create the input to the BERT Next Sentence Prediction Model (run_nsp.py)

`src/run_nsp.py`: BERT finetuning runner for Next Sentence Prediction
```
usage: run_nsp.py [-h] --req_pretrained REQ_PRETRAINED --model_dir MODEL_DIR --data_dir DATA_DIR
                  --vocab_dir VOCAB_DIR --task_name TASK_NAME --output_dir
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

Note: req_pretrained is used when we are randomly initializing the Next Sentence Prediction head i.e. when we use only the SciBERT model alone wihout finetuning on the NSP Task
In technical terms, it tells the code to load the model using the from_pretrained method
```

`Run_BERT_COVID_NSP.sh`: Bash Script to execute run_nsp.py using GPU's. \
```usage: bash Run_BERT_COVID_NSP.sh ``` or ```sbatch Run_BERT_COVID_NSP.sh (for SLURM Support)``` 

`src/analyze_nsp_results.py`: Code to read the output csv from run_nsp.py and metadata.csv. It filters papers based on a threshold set as 0.999 for
either Treatment or Vaccine Queries. It assigns label based on whichever is higher beyond 0.999 and O if it isn't beyond
the threshold.

`src/convert_csv_results_json.py`: Code to convert the CSV formatted results to JSON for faster retrieval during Ensemble

`src/lm_finetuning/`: Contains code to do MLM/NSP Intermediate FineTuning
