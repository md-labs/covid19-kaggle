"""
Code to combine all text which is classified as Vaccine or Therapeutics by the Query-Document NSP Model and prepare
input to the Clinical Hedges classifier (run_classifier.py)
Input: Directory of filtered results from NSP model
Output: Data prepped in format required for input to Clinical Hedges Classification Model
Output File Header Format: ["ID", "Text", "Label", "Prob"]
"""

import csv
import os

path_to_results = os.path.abspath('../../document_nsp_text_sim/results/filtered_results_final')
dirListing = os.listdir(path_to_results)

combined_data_dict = dict()
for file in dirListing:
    with open(os.path.join(path_to_results, file)) as fp:
        reader = csv.reader(fp, delimiter='\t')
        for i, row in enumerate(reader):
            if i == 0:
                continue
            combined_data_dict[row[0]] = row[1:]


with open(os.path.abspath("../data/Pred_Data_COVID/Test_Data.tsv"), 'w') as fp:
    writer = csv.writer(fp, delimiter='\t')
    writer.writerow(["ID", "Text", "Label", "Prob"])
    for key in combined_data_dict.keys():
        if combined_data_dict[key][1] == 'O':
            continue
        writer.writerow([key] + combined_data_dict[key])
