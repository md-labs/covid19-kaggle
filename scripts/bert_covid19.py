# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
import unidecode
import re
from sklearn.model_selection import train_test_split
#from statistics import mean

import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.autograd import Variable
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm, trange
from scipy.stats.stats import pearsonr

from transformers import BertTokenizer, BertConfig
from transformers import AdamW, BertModel
from transformers import *

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def create_input_features(df, max_length, bert_model):
    
    logger.info("Shape of data = %s", str(df.shape))

    # Create sentence and label lists

    sentences_1 = df.sentence1.values
    sentences_2 = df.sentence2.values
    #similarity_score = df.score.values
    
    # We need to add special tokens at the beginning and end of each sentence for BERT to work properly

    special_sentences_tempe_1 = ["[CLS] " + sentence for sentence in sentences_1]
    special_sentences_tempe_2 = [" [SEP] " + sentence for sentence in sentences_2]
    special_sentences_1 = [i + j for i, j in zip(special_sentences_tempe_1, special_sentences_tempe_2)]
    
    special_sentences= []
    for sentence in special_sentences_1:
        if len(sentence)>max_length:
            sentence= sentence[0:512]
        special_sentences.append(sentence)
    
    logger.info("Step 1 done: creating Special sentences...")
    
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case= False)
    tokenized_texts = [tokenizer.tokenize(sentence) for sentence in special_sentences]
    logger.info("Step 2 done: creating tokens from step 1...")
    
    # Max sentence input 
    MAX_LEN = max_length
    
    # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
    input_sentences = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

    # Pad our input tokens
    input_sentences = pad_sequences(input_sentences, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # Create attention Masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding

    for seq in input_sentences:
      seq_mask = [float(i>0) for i in seq]
      attention_masks.append(seq_mask)
  
    logger.info("Step 3 done: creating Attention Masks...")
    
    return input_sentences, attention_masks

def creating_testdata(query, data_path):
    
    test_path= data_path + '/paper_data_up.tsv'
    df = pd.read_csv(test_path, sep='\t')

    df['sentence1']= query
    
    with open(join(data_path,'test.tsv'),'w') as write_tsv:
        write_tsv.write(df.to_csv(sep='\t', index=False))
        
def mean(result):
    return sum(result)/len(result)
    

class linearRegression(nn.Module):
    def __init__(self):
        super(linearRegression, self).__init__()
        self.linear = nn.Linear(768, 1)  # input and output is 1 dimension

    def forward(self, x):
        out = self.linear(x)
        return out


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default="",
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default="", type=str, 
                        required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default="",
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--query",
                        default="Effectiveness of drugs like naproxen, clarithromycin, and minocyclinethat being developed that may exert effects on viral replication and tried to treat COVID-19 patients",
                        type=str,
                        help="Put you query here")
    
    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    
    args = parser.parse_args()
    
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    
    if args.do_train:
        logger.info("Creating Training Data")
        train_path= args.data_dir+'/train.tsv'
        df_train= pd.read_csv(train_path, sep='\t')
        input_sentences_train, attention_masks_train= create_input_features(df_train, args.max_seq_length, args.bert_model)
        score_train= df_train.score
        
        logger.info("Creating Dev Data")
        dev_path= args.data_dir+'/dev.tsv'
        df_dev= pd.read_csv(dev_path, sep='\t')
        input_sentences_dev, attention_masks_dev= create_input_features(df_dev, args.max_seq_length, args.bert_model)
        score_dev= df_dev.score
        
        # Convert all of our data into torch tensors, the required datatype for our model
        
        train_inputs = torch.tensor(input_sentences_train)
        dev_inputs = torch.tensor(input_sentences_dev)
        train_labels = torch.tensor(score_train)
        dev_labels = torch.tensor(score_dev)
        train_masks = torch.tensor(attention_masks_train)
        dev_masks = torch.tensor(attention_masks_dev)
        
        # Select a batch size for training. For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32
        batch_size = args.train_batch_size
        
        # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
        # with an iterator the entire dataset does not need to be loaded into memory
        
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
        
        dev_data = TensorDataset(dev_inputs, dev_masks, dev_labels)
        dev_sampler = SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=batch_size)
        
        logger.info("#------------------------------------------------------#")
        logger.info("All Data Processing Tasks are Done for Training")
        logger.info("#------------------------------------------------------#")
    
    ## Loading BERT Model
    
    model = BertModel.from_pretrained(args.bert_model)
    model_class= BertModel
    model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
    optimizer = AdamW(optimizer_grouped_parameters,lr=args.learning_rate)
    
    regression = linearRegression()
    regression.to(device)
    criterion = nn.MSELoss()
    optimizer_reg = torch.optim.SGD(regression.parameters(), lr=1e-4)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if args.do_train:
        for ep in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            regression.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, similarity_score = batch
                optimizer_reg.zero_grad()
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=input_mask)
                last_hidden_states = outputs[1]
                pred_score= regression(last_hidden_states)
                pred_score= np.squeeze(pred_score, axis=1)
                loss= criterion(pred_score, similarity_score)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                
                loss.backward()
                optimizer_reg.step()
                optimizer.step()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                global_step += 1
            
            print("\n")
            print("Running evaluation for epoch: {}".format(ep))
            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            for step, batch in enumerate(dev_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, similarity_score= batch
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                similarity_score = similarity_score.to(device)
                    
                outputs = model(input_ids, attention_mask=input_mask)
                last_hidden_states = outputs[1]
                pred_score= regression(last_hidden_states)
                pred_score= np.squeeze(pred_score, axis=1)
                tmp_eval_loss= criterion(pred_score, similarity_score)
                
                pred_score = pred_score.detach().cpu().numpy()
                similarity_score = similarity_score.detach().cpu().numpy()
                tmp_eval_accuracy = pearsonr(pred_score, similarity_score)[0]

                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy

                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_steps
            loss = tr_loss/nb_tr_steps if args.do_train else None
            result = {'eval_loss': eval_loss,
                      'eval_accuracy': eval_accuracy,
                      'global_step': global_step,
                      'loss': loss}

            for key in sorted(result.keys()):
                print(key, str(result[key]))
            print()
            	
    if args.do_train:
        output_dir= args.output_dir
        # Save a trained model and the associated configuration
        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Saving model to %s" % output_dir)

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        torch.save(regression, join(output_dir,"regression_model.pth"))

    ## Loading BERT Model

    if args.do_eval:
        #model_class= BertModel
        output_dir= args.output_dir 
        logger.info("***** Running Testing *****")
        # Load a trained model and config that you have fine-tuned
        model = model_class.from_pretrained(output_dir)
        regression = torch.load(join(output_dir,"regression_model.pth"))
        model.to(device)
        regression.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        model.eval()
        regression.eval()
        
        test_path= args.data_dir + '/test.tsv'
        df_test= pd.read_csv(test_path, sep='\t')
        abstracts= df_test.sentence
        
        final_result_max= list()
        final_result_mean= list()
        final_result_5mean= list()
        final_result_3mean= list()
        count=1
        
        for abstract in abstracts:
            iteration = int(len(abstract)/args.max_seq_length)
            
            if iteration==0:
                final_result_max.append(0)
                final_result_mean.append(0)
                final_result_5mean.append(0)
                final_result_3mean.append(0)
                count+=1
                continue
            
            result= list()
            sentence= list()
            
            for i in range(0,iteration):
                sent= abstract[(i*args.max_seq_length):(i+1)*args.max_seq_length]
                sentence.append(sent)
            
            df_temp= pd.DataFrame(sentence, columns=['sentence2'])
            df_temp['sentence1']= args.query
            
            input_sentences, attention_masks= create_input_features(df_temp, args.max_seq_length, args.bert_model)
            # Convert all of our data into torch tensors, the required datatype for our model

            test_inputs = torch.tensor(input_sentences)
            test_masks = torch.tensor(attention_masks)

            # Select a batch size for training. For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32
            batch_size = args.eval_batch_size

            # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
            # with an iterator the entire dataset does not need to be loaded into memory

            test_data = TensorDataset(test_inputs, test_masks)
            test_sampler = RandomSampler(test_data)
            test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
            
            for step, batch in enumerate(test_dataloader):
                input_ids, input_mask= batch
                input_ids=input_ids.to(device)
                input_mask=input_mask.to(device)
                
                outputs = model(input_ids, attention_mask=input_mask)
                last_hidden_states = outputs[1]
                pred_score= regression(last_hidden_states)
                pred_score= np.squeeze(pred_score, axis=1)
                pred_score = pred_score.detach().cpu().numpy()
                result.extend(pred_score)
            
            result.sort(reverse=True)
            result_5=result[0:5]
            result_3=result[0:3]
            final_result_max.append(max(result))
            final_result_mean.append(mean(result))
            final_result_5mean.append(mean(result_5))
            final_result_3mean.append(mean(result_3))
            print("#------------------------------------------------------#")
            print("Done Testing " + str(count))
            print("#------------------------------------------------------#")
            count+=1
            
        data = list(zip(final_result_max, final_result_mean, final_result_5mean, final_result_3mean))
        df_result= pd.DataFrame(data, columns=['max','mean', 'mean5', 'mean3'])
        df_result.to_csv('./result.csv')
        df_test['max']= final_result_max
        df_test['mean']= final_result_mean
        df_test['mean5']= final_result_5mean
        df_test['mean3']= final_result_3mean
        
        df_test.to_csv('./final_result.csv')

        logger.info("#------------------------------------------------------#")
        logger.info("Done All Testing")
        logger.info("#------------------------------------------------------#")
            
if __name__ == "__main__":
    main()
