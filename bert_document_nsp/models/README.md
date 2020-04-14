# Models for bert_document_level_nsp

The SCIBERT model from AllenAI is used in four different variants:
1. Pretrained SciBERT model with NSP (without fine tuning)
2. Fine Tuned SciBERT Model using MLM on the Abstract / Title Text of the COVID Dataset and then use this for NSP (uses FineTune.txt)
3. Fine Tuning the Pretrained SciBERT model in variant 1 with the Opioid NSP Question-Answer dataset and then use this model for NSP
4. Fine Tuning the Fine-Tuned SciBERT model in variant 2 with the Opioid NSP Question-Answer dataset and then use this model for NSP

Each of these models can be found [here](https://www.dropbox.com/sh/ko0d8jayaapb7xq/AABZ1yPVCLFuKUrPoBXBfjD0a?dl=0)

Labels for the folders are self explanatory about which models depict which variant
