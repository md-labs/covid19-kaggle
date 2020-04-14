"""
Code to read the output csv from run_nsp.py and metadata.csv. It filters papers based on a threshold set as 0.999 for
either Treatment or Vaccine Queries. It assigns label based on whichever is higher beyond 0.999 and O if it isn't beyond
the threshold.
Input: metadata.csv, Reqd_Labels.csv from run_nsp.py
Output: Threshold Filtered Results formatted as 'Title', 'Text', 'Label', 'Prob_T', 'Prob_V'
"""


from bert_document_nsp.utils import ReadData
import os
import csv
import sys

path_to_data = os.path.abspath("../data")
path_to_results = os.path.abspath("../results")
csv.field_size_limit(sys.maxsize)


def main():
    RetCovData = ReadData(path_to_data + '/raw_data', "metadata.csv")
    RetCovDict = dict()
    text = []
    for i, row in enumerate(RetCovData):
        if i == 0:
            continue
        RetCovDict[row[3]] = ' '.join((row[3] + ' ' + row[8] + ' ' + row[11]).split('\n'))
        text.append(' '.join((row[3] + ' ' + row[8] + ' ' + row[11]).split('\n')))
    labels = ReadData(os.path.join(path_to_results, "nsp_results_final"), "Reqd_Labels_Before_FineTuning.csv")
    label_dict = dict()
    for i, row in enumerate(labels):
        if i == 0:
            continue
        if row[0][:-1] not in label_dict:
            label_dict[row[0][:-1]] = dict()
        label_dict[row[0][:-1]][row[0][-1]] = float(row[1])
    count = 0
    final_labels = []
    for key in label_dict.keys():
        if key == 'paper_id':
            continue
        dictionary = label_dict[key]
        if 'T' not in dictionary or 'V' not in dictionary:
            continue
        th = 0.999
        if dictionary['T'] > th or dictionary['V'] > th:
            label = 'TR' if dictionary['T'] > dictionary['V'] else 'VC'
            final_labels.append([key, RetCovDict[key], label, dictionary['T'], dictionary['V']])
            count += 1
        else:
            final_labels.append([key, RetCovDict[key], 'O', dictionary['T'], dictionary['V']])
    print(count)
    final_labels = list(sorted(final_labels, key=lambda x: x[3], reverse=True))
    with open(os.path.join(path_to_results, 'filtered_results_final', 'Filter_Before_FineTuning.tsv'), 'w', newline='',
              encoding='utf-8') as fp:
        writer = csv.writer(fp, delimiter='\t')
        writer.writerow(['Title', 'Text', 'Label', 'Prob_T', 'Prob_V'])
        for row in final_labels:
            writer.writerow(row)


if __name__ == '__main__':
    main()
