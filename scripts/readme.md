# COVID-19: BERT-based STS Method for Searching Relevant Research Papers

The objective of this project is to use the state-of-the-art Sementic Textual Similarity (STS) technique to create a method to search a answer for the given query through the dataset of research papers provided as a part of Kaggle's competion CORD-19-research-challenge. Currently, we are considering limited amount of given text (Titles + Abstracts + Journals) for each paper to show the effectiveness of the proposed approach. Gradually, this project could be extended into scientific search system where you can extract machine readable scientific data for given query for further analysis.

### Setup Environment

1. python3.6 <br /> Reference to download and install : https://www.python.org/downloads/release/python-360/
2. Install requirements <br /> 
```
$ pip3 install -r requirements.txt
```

## Run the GA
1. Use default setting as given below:
```
--data_dir=./data 
--bert_model=bert-base-uncased 
--output_dir=./output_128 
--max_seq_length=128 
--num_train_epochs=10 
--do_eval 
--train_batch_size=8 
--eval_batch_size=1
```
To run algorithm on Agave GPU cluster, use command:
```
$ sh agave_run.sh
```

2. To change the parameter setting, you need to use command:

```
$ python3 bert_covid.py --data_dir=./data [path of folder where training and testing data is located] 
--bert_model=bert-base-uncased [Type of bert model you want to use]
--output_dir=./output [path where you want to save the results]
--max_seq_length=128 [Max length for BERT]
--num_train_epochs=10 [Number of training epochs (int)]
--do_train [for training]
--do_eval [for testing] 
--train_batch_size=8 [batch size for training (int)]
--eval_batch_size=1 [batch size for testing (int)]
``` 
