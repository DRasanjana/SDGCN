# -*- coding: utf-8 -*-
'''
# @Author  : plzhao
# @Software: PyCharm
'''
from bert_embedding import BertEmbedding
import numpy as np
import time
import string
import re
import csv

def remove_punct(text):
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table)

def load_data_and_labels(positive_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files

    with open(positive_data_file, 'r', encoding="utf8") as csvfile:
        aspectreader = csv.reader(csvfile, delimiter=',')
        j = 0
        count = 0
        input = []
        target = []
        lable = []
        for row in aspectreader:
            if (j == 0):
                j = 1
            else:
                sent = row[0].lower()
                sent = remove_punct(sent)
                sent.replace('\d+', '')
                # sent.replace(r'\b\w\b', '').replace(r'\s+', ' ')
                # sent.replace('\s+', ' ', regex=True)
                # sent=re.sub(r"^\s+|\s+$", "", sent), sep='')
                sent = re.sub(r"^\s+|\s+$", "", sent)
                input.append(sent)
                # nb_aspects = int(row[1])
                aspect = row[1].lower()
                target.append(aspect)
                polarity = row[2]
                if int(polarity)==1:
                    lable.append([1, 0, 0])
                elif int(polarity) == 0:
                    lable.append([0, 1, 0])
                elif int(polarity) == -1:
                    lable.append([0, 0, 1])
        x_text = input
        y = np.array(lable)
        return [x_text, target, y]

def load_targets(positive_data_file):
    """
    find the same sentence,output all the targets of each sentence.
    output the targets' number of each sentences
    """
    # Load data from files
    with open(positive_data_file, 'r', encoding="utf8") as csvfile:
        aspectreader = csv.reader(csvfile, delimiter=',')
        j = 0
        count = 0
        input = []
        target = []
        lable = []
        examples=[]
        ccc=[]
        for row in aspectreader:
            if (j == 0):
                j = 1
            else:
                sent = row[0].lower()
                sent = remove_punct(sent)
                sent.replace('\d+', '')
                # sent.replace(r'\b\w\b', '').replace(r'\s+', ' ')
                # sent.replace('\s+', ' ', regex=True)
                # sent=re.sub(r"^\s+|\s+$", "", sent), sep='')
                sent = re.sub(r"^\s+|\s+$", "", sent)
                examples.append(sent)
                input.append(sent)
                # nb_aspects = int(row[1])
                aspect = row[1].lower()
                target.append(aspect)
                examples.append(aspect)
                polarity = row[2]
                examples.append(polarity)
                ccc.append(sent+","+aspect+","+polarity+","+row[3]+","+row[4])
    x_text = input
    # find the same targets
    all_sentence = [s for s in x_text]
    targets_nums = [all_sentence.count(s) for s in all_sentence]
    # ccc_num= [ccc.count(s) for s in ccc]
    # for i in range(len(ccc_num)):
    #     if ccc_num[i]>1:
    #         print(str(i+2) +ccc[i])

    # i=0
    # while i<len(targets_nums):
    #     act=int(targets_nums[i])
    #     for j in range(act):
    #         if not (targets_nums[i+j]==act):
    #             print(x_text[i+j-1])
    #             print(targets_nums[i+j])
    #             print(i+j-1)
    #     i=i+j+1


    targets = []
    i = 0
    while i < len(all_sentence):
        num = targets_nums[i]
        target = []
        for j in range(num):
            target.append(examples[(i + j) * 3 + 1])
        for j in range(num):
            targets.append(target)
        i = i + num
    targets_nums = np.array(targets_nums)
    return [targets, targets_nums]


    # examples = list(open(positive_data_file, "r").readlines())
    # examples = [s.strip() for s in examples]
    #
    # input = []
    # target = []
    # for index,i in enumerate(examples):
    #     if index%3 == 0:
    #         i_target =examples[index + 1].strip()
    #         i = i.replace("$T$", i_target)
    #         input.append(i)
    #         target.append(i_target)
    # x_text = input
    # # find the same targets
    # all_sentence = [s for s in x_text]
    # targets_nums = [all_sentence.count(s) for s in all_sentence]
    # targets = []
    # i = 0
    # while i < len(all_sentence):
    #     num = targets_nums[i]
    #     target = []
    #     for j in range(num):
    #         target.append(examples[(i+j)*3+1])
    #     for j in range(num):
    #         targets.append(target)
    #     i = i+num
    # targets_nums = np.array(targets_nums)
    # return [targets,targets_nums]


def get_targets_array(target_array,targets_num,max_target_num):
    """
    结合输入的target_position以及target_num,target_num是多少，就由多少个，并且重复多少次。
    不足max_target_num的，补0.
    """
    positions = []
    i = 0
    while i < targets_num.shape[0] :
        i_position = []
        for t_num in range(targets_num[i]):
            i_position.append(target_array[i+t_num])

        for j in range(max_target_num - targets_num[i]):
            i_position.append(np.zeros([target_array.shape[1],target_array.shape[2]]))
        for t_num in range(targets_num[i]):
            positions.append(i_position)
            i += 1

    return np.array(positions)



#-----------------------Restaurants--------------------------
print('-----------------------Restaurants--------------------------')
train_file = "data_res/train.csv"
test_file = "data_res/test.csv"

train_target_load_file = "data_res/Train_target_Embedding.npy"
test_target_load_file = "data_res/Test_target_Embedding.npy"
train_targets_save_file = "data_res/Train_targets_Embedding.npy"
test_target_save_file = "data_res/Test_targets_Embedding.npy"

print("loading data:")

train_targets_str, train_targets_num = load_targets(train_file)
test_targets_str, test_targets_num = load_targets(test_file)
max_target_num = max([len(x) for x in (train_targets_str + test_targets_str)])

train_target_array = np.load(train_target_load_file)
test_target_array = np.load(test_target_load_file)      #([1120,23,768])
train_targets_array = get_targets_array(train_target_array,train_targets_num,max_target_num)
test_targets_array = get_targets_array(test_target_array,test_targets_num,max_target_num)       #([1120,13,23,768])

np.save(train_targets_save_file,train_targets_array)
np.save(test_target_save_file,test_targets_array)
print("finish save --targets array-- in: ", train_targets_save_file)
print("finish save --targets array-- in: ", test_target_save_file)
print()





