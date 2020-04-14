# Data used for bert_document_level_nsp

raw_data: [Link](https://www.dropbox.com/sh/g6zbetusch4m5l6/AAD7DG1s44ZHxdPzb6x_TW2wa?dl=0)

NSP_Data: [Link](https://www.dropbox.com/sh/ddrm75oofwx0qt1/AABO9-pTw4TxvtSd51eylwsMa?dl=0)

In the NSP_Data folder, 

`FineTune.txt`: lists sentences from corpus with blank lines between two sentences signifying two different documents. This is used for FineTuning the model using the Masked Language Model and Next Sentence Prediction Training Methods employed in the BERT paper. Intermediate FineTuning seems to help as mentioned in this work: [ULMFiT paper](https://arxiv.org/abs/1801.06146)

`COVID_NSP_DATA`: Contains the COVID Data to Predict Similarity in the format of [CLS] Query [SEP] Text [SEP]

`OPIOID_NSP_DATA`: Contains data scraped from Reddit Opioid Forums and maunally annotated question/answers. Used as fine tuning data for NSP.

`vaccine_and_therapeutics_query.json`: Contains the Manually Created Vaccine and Therapeutics Query handpicked from Clinical Papers.  