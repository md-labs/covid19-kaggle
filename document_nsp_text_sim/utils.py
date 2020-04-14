"""
Utility code to do simple functions like read. shuffle and write data.
"""

import csv
import os
import random


def ShuffleData(data):
    index = []
    for i in range(len(data)):
        index.append(i)
    req_index = random.sample(index, len(data))
    shuffled_data = []
    for index in req_index:
        shuffled_data.append(data[index])
    return shuffled_data


def ReadData(path, file_name):
    RetList = []
    with open(os.path.join(path, file_name), encoding='utf8') as fp:
        reader = csv.reader(fp)
        for row in reader:
            RetList.append(row)
    return RetList


def WriteData(file_name, data, path):
    with open(os.path.join(path, file_name), 'w', newline='', encoding='utf-8') as fp:
        writer = csv.writer(fp, delimiter='\t')
        writer.writerow(["ID", "Text", "Label", "Task_Number"])
        for row in data:
            writer.writerow(row)