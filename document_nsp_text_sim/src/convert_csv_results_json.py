"""
Code to convert the CSV formatted results to JSON for faster retrieval during Ensemble
Input: CSV with header: 'Title', 'Text', 'Label', 'Prob_T', 'Prob_V'
Output: JSON with key as Title and Value as [Prob_T, Prob_V, Prob_O, Label]
"""

import json
import csv
import os

ch_classification = False
results = []
path_to_results = os.path.abspath('../results')
with open(os.path.join(path_to_results, 'threhold_filtered_results', 'Filter_After_FineTuning.tsv')) as fp:
    reader = csv.reader(fp, delimiter='\t')
    for i, row in enumerate(reader):
        if i == 0:
            continue
        results.append(row)

result_json = dict()

for row in results:
    row[3] = float(row[3])
    if not ch_classification:
        row[4] = float(row[4])

    if row[2] == 'O':
        maximum_index = 3 if row[3] == max(row[3], row[4]) else 4
        if maximum_index == 3:
            result_json[row[0]] = [1 - row[3], row[4], row[3], row[2]]
        else:
            result_json[row[0]] = [row[3], 1 - row[4], row[4], row[2]]
    else:
        if ch_classification:
            result_json[row[0]] = [row[3], 1-row[3], 0, row[2]]
        else:
            result_json[row[0]] = [row[3], row[4], 1 - max(row[3], row[4]), row[2]]

with open(os.path.join(path_to_results, 'final_json_results', 'Filter_After_FineTuning.json'), 'w') as fp:
    json.dump(result_json, fp)
