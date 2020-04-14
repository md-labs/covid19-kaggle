"""
Code that reads the predicted output from the Clinical Hedges model and writes it along with the Text as output
Input: Label output (csv) from the classification model and metadata.csv
Output: Data Labels along from the label output file along with its corresponding text from the metadata file
"""

from document_nsp_text_sim.utils import ReadData
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
        text.append(' '.join((row[2] + " " + row[3]).split('\n')))
    labels = ReadData(os.path.join(path_to_results, "Classification_Reports_CH", "Reqd_Labels_COVID"),
                      "Reqd_Labels_Task_3.csv")
    label_dict = dict()
    for i, row in enumerate(labels):
        if i == 0:
            continue
        row[0] = row[0].replace("\"", "")
        label_dict[row[0]] = [row[2], row[3]]
    final_labels = []
    for key in label_dict.keys():
        temp_list = label_dict[key]
        label = temp_list[0]
        prob = float(temp_list[1])
        final_labels.append([key, RetCovDict[key], label, prob])
    final_labels = list(sorted(final_labels, key=lambda x: x[3], reverse=True))
    with open(os.path.join(path_to_results, 'Classification_Reports_CH', 'COVID_Labels_Final',
                           'COVID_Labels_After_Classification_CH.tsv'), 'w',
              newline='', encoding='utf-8') as fp:
        writer = csv.writer(fp, delimiter='\t')
        writer.writerow(['ID', 'Text', 'Label', 'Prob'])
        for row in final_labels:
            writer.writerow(row)


if __name__ == '__main__':
    main()
