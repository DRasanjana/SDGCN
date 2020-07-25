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

def create_bert_embedding(input,max_len):
    print("creating BERT embedding ")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    result = bert_embedding(input)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("finish BERT embedding ")
    # padding
    sentence_BERT = []
    for i in result:
        embedding_i = i[1]  #句子长度的list，每一个元素都是一个词向量。
        pad = [np.zeros(768)]
        sentence_BERT_i = embedding_i + pad * (max_len - len(i[0]))
        sentence_BERT.append(sentence_BERT_i)

    return np.array(sentence_BERT)

def save_BERT_embeddinf(save_file,bert_embedding):
    np.save(save_file, bert_embedding)
    print("Finish save BERT embedding in: ", save_file, "\n")
    print()

train_file = "data_res/train.csv"
test_file = "data_res/test.csv"
train_save_file = "data_res/Train_Embedding.npy"
test_save_file = "data_res/Test_Embedding.npy"
train_target_save_file = "data_res/Train_target_Embedding.npy"
test_target_save_file = "data_res/Test_target_Embedding.npy"


bert_embedding = BertEmbedding(model='bert_12_768_12', dataset_name='book_corpus_wiki_en_uncased',max_seq_length=100)
# result2 = bert_embedding(["eat apple","apple tree"])
print("loading data:")
train_x_str, train_target_str, train_y = load_data_and_labels(train_file)
test_x_str, test_target_str, test_y = load_data_and_labels(test_file)
max_sentence_length = max([len(x.split(" ")) for x in (train_x_str + test_x_str)])
max_target_length = max([len(x.split(" ")) for x in (train_target_str + test_target_str)])

#create_bert_embedding
train_BERT_em = create_bert_embedding(train_x_str, max_sentence_length)
test_BERT_em = create_bert_embedding(test_x_str, max_sentence_length)

train_target_BERT_em = create_bert_embedding(train_target_str, max_target_length)
test_target_BERT_em = create_bert_embedding(test_target_str, max_target_length)

#save_BERT_embeddinf
save_BERT_embeddinf(train_save_file,train_BERT_em)
save_BERT_embeddinf(test_save_file,test_BERT_em)
save_BERT_embeddinf(train_target_save_file,train_target_BERT_em)
save_BERT_embeddinf(test_target_save_file,test_target_BERT_em)
