"""
Code to create the input to the BERT Next Sentence Prediction Model (run_nsp.py)
Input:  vaccine_and_therapeutics_query.json, metadata.csv
Output: NSP Formatted Data in the form: [CLS] Query [SEP] (Title + Abstract) Text [SEP]
"""


import os
import json
import csv
import shutil
import sys
from document_nsp_text_sim.utils import ReadData

csv.field_size_limit(sys.maxsize)

path_to_data = os.path.abspath("../data")


def WriteDataNSP(dataset, path, directory):
    if os.path.exists(os.path.join(path_to_data, directory)):
        shutil.rmtree(os.path.join(path_to_data, directory))
    os.mkdir(os.path.join(path_to_data, directory))
    with open(os.path.join(path, directory, 'Test_Data.tsv'), 'w', newline='', encoding='utf-8') as fp:
        writer = csv.writer(fp, delimiter='\t')
        writer.writerow(['ID', 'Text_A', 'Text_B', 'DEF'])
        for row in dataset:
            writer.writerow(row)

def main():
    RetCovData = ReadData(path_to_data + '/raw_data', "metadata.csv")
    vctdict = json.load(open(os.path.join(path_to_data, "vaccine_and_therapeutics_query.json")))
    dataset = []
    for i, row in enumerate(RetCovData):
        if i == 0:
            continue
        if row[2] == '' or row[3] == '':
            continue
        dataset.append([row[3] + 'V', vctdict['vaccine'], ' '.join((row[3] + ' ' + row[8] + ' ' + row[11]).split('\n')),
                        'VC'])
        dataset.append([row[3] + 'T', vctdict['therapeutics'], ' '.join((row[3] + ' ' + row[8] + ' ' + row[11]).split('\n')),
                        'TR'])
    WriteDataNSP(dataset, path_to_data, 'COVID_NSP_Data')


if __name__ == '__main__':
    main()
