import numpy as np
import os
import re
import itertools
import json
import requests
import scipy.sparse as sp
import pickle
from collections import Counter
from nltk.corpus import stopwords
from tqdm import tqdm
import ast

from hybrid_xml import arr_length

cachedStopWords = stopwords.words("english")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit, ShuffleSplit

def clean_str(string):
    # remove stopwords
    # string = ' '.join([word for word in string.split() if word not in cachedStopWords])
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def pad_sentences(sentences, padding_word="<PAD/>", max_length=500):
    sequence_length = min(max(len(x) for x in sentences), max_length)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        if len(sentence) < max_length:
            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
        else:
            new_sentence = sentence[:max_length]
        padded_sentences.append(new_sentence)
    return padded_sentences


def load_data_and_labels(data):
    x_text = [clean_str(doc['text']) for doc in data]
    x_text = [s.split(" ") for s in x_text]
    # labels = [doc['catgy'] for doc in data]
    labels = []
    # for doc in data:
    #     labels.append([tuple(tuple(a) for a in doc['catgy'])])

    # labels =  [doc['catgy'][0] +[x + 19 for x in doc['catgy'][1]]  for doc in data] # add 18 to the second object to have a 1d array
    # for label in labels:
    labels = []
    for doc in data:
        tmp_list = []
        # tmp_list =doc['catgy'][0]
        # tmp_list.append(18)
        for i in range(len(doc['catgy'])):
            tmp_list += [x + 19 * (i) for x in doc['catgy'][i]]
            # tmp_list.append(doc['catgy'][i]+19*i)
            tmp_list.append(18 + 19 * (i))

        labels.append(tmp_list)

    row_idx, col_idx, val_idx = [], [], []
    for i in tqdm(range(len(labels))):
        l_list = list(set(labels[i]))  # remove duplicate cateories to avoid double count
        # l_list = labels[i]  # remove duplicate cateories to avoid double count
        # l_list = {tuple(i) for i in labels}
        for y in l_list:
            row_idx.append(i)
            col_idx.append(y)
            val_idx.append(1)

    m = max(row_idx) + 1
    n = max(col_idx) + 1
    n= arr_length # TODO: make this adaptive
    # n = max([max(x) for x in col_idx])
    Y = sp.csr_matrix((val_idx, (row_idx, col_idx)), shape=(m, n))
    # Y = col_idx

    return [x_text, Y]


def load_data_and_labels_n(data):
    x_text = [clean_str(doc['text']) for doc in data]
    x_text = [s.split(" ") for s in x_text]
    labels = [doc['catgy'] for doc in data]

    row_idx, col_idx, val_idx = [], [], []
    for i in tqdm(range(len(labels))):
        l_list = list(set(labels[i][0]))  # remove duplicate cateories to avoid double count
        for y in l_list:
            row_idx.append(i)
            col_idx.append(y)
            val_idx.append(1)

    m = max(row_idx) + 1
    n = max(col_idx) + 1
    n = arr_length  # TODO: make this adaptive
    # n = max([max(x) for x in col_idx])
    Y = sp.csr_matrix((val_idx, (row_idx, col_idx)), shape=(m, n))
    # Y = col_idx

    return [x_text, Y]
def build_vocab(sentences, vocab_size=50000):
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [x[0] for x in word_counts.most_common(vocab_size)]
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    # append <UNK/> symbol to the vocabulary
    vocabulary['<UNK/>'] = len(vocabulary)
    vocabulary_inv.append('<UNK/>')
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, vocabulary):
    x = np.array(
        [[vocabulary[word] if word in vocabulary else vocabulary['<UNK/>'] for word in sentence] for sentence in
         sentences])
    # x = np.array([[vocabulary[word] if word in vocabulary else len(vocabulary) for word in sentence] for sentence in sentences])
    return x


def get_valid_df():
    test_data = []
    #elements = requests.get(
    #     "https://api.baserow.io/api/database/rows/table/17789/",
    #     headers={
    #         "Authorization": "Token RGCVcpkyOlPeEpsfLg0z2zjaG6TyUbGx"
    #     }
    # )
    #data = elements.json()
    f = open("dataset_baserow.json", "r")
    data = json.loads(f.read())

    for element in data['results']:
        filename = element['field_93157'].split(':')[0]
        inner_data = ast.literal_eval(element['field_93158'])
        description = inner_data['description']
        if len(description)<10 or 'computational'  in description:
            continue

        if True in [description in dicti.values() for dicti in test_data]:
            continue
        given_exp = ast.literal_eval(inner_data['given_exp'])
        if len(given_exp[1])==18:
            continue
        true_exp = ast.literal_eval(inner_data['true_exp'])[1]
        categories = []
        for obj in true_exp:
            obj = obj + [True]
            categories.append([i for i, x in enumerate(obj) if x])

        test_data.append({'text':description,'Id':filename,'split':'val',
                          'catgy':categories,  'num_words': len(description)
                          })



    return test_data


def load_data(data_path, max_length=500, vocab_size=50000, split=0):
    with open(os.path.join(data_path), 'rb') as fin:
        # load data
        # df = pd.read_json("../MasterThesis/one_to_4_25noise_shuffled order.json")
        df = pd.read_json(data_path)
        # df = pd.read_json("../MasterThesis/two_objects.json")
        # df = pd.read_json("../MasterThesis/edited_files/all_edited.json")

    df.head()
    df['labels'] = df[df.columns[2:]].values.tolist()
    new_df = df[['description', 'solution_matrix', 'file_name']].copy()
    new_df.head()
    #  [train, test, vocab, catgy] =  []
    # split train_val and test
    sss = ShuffleSplit(n_splits=5, test_size=0.3, random_state=5)

    splits = [(train, test) for train, test in sss.split(new_df.description, new_df.solution_matrix)]
    train_val_index, test_index = splits[split]

    splits = [(train, test) for train, test in sss.split(new_df.description.iloc[train_val_index],
                                                         new_df.solution_matrix.iloc[train_val_index])]

    tmp_idx_train, tmp_idx_val = splits[0]
    train_index = train_val_index[tmp_idx_train]
    val_index = train_val_index[tmp_idx_val]

    train_df = new_df.iloc[train_index].reset_index(drop=True)
    test_df = new_df.iloc[test_index].reset_index(drop=True)

    valid = get_valid_df()
    # train_df, test_df = train_test_split(new_df, test_size=0.2)
    # now turn the df into arrays with dicts : 'split', 'text' 'Id'  'catgy'(list of label inices' 'num_words'
    train = []
    test = []
    for index, row in train_df.iterrows():
        # categories = [i for i, x in enumerate(row['solution_matrix']) if x]
        categories = []
        for obj in row['solution_matrix']:
            categories.append([i for i, x in enumerate(obj) if x])
        train.append({'split': 'train', 'text': row['description'], 'Id': row['file_name'], 'catgy': categories,
                      'num_words': len(row['description'])})
    for index, row in test_df.iterrows():
        categories = []
        for obj in row['solution_matrix']:
            categories.append([i for i, x in enumerate(obj) if x])

        test.append(
            {'split': 'test', 'text': row['description'], 'Id': row['file_name'], 'catgy': categories,
             'num_words': len(row['description'])})

    if len(test) == 0:
        test[:5] = train[:5]
    trn_sents, Y_trn = load_data_and_labels(train)
    tst_sents, Y_tst = load_data_and_labels(test)
    val_sents, Y_val = load_data_and_labels(valid)


    trn_sents_padded = pad_sentences(trn_sents, max_length=max_length)
    tst_sents_padded = pad_sentences(tst_sents, max_length=max_length)
    val_sents_padded = pad_sentences(val_sents, max_length=max_length)
    print("len:", len(trn_sents_padded), len(tst_sents_padded))
    vocabulary, vocabulary_inv = build_vocab(trn_sents_padded + tst_sents_padded+val_sents_padded, vocab_size=vocab_size)
    X_trn = build_input_data(trn_sents_padded, vocabulary)
    X_tst = build_input_data(tst_sents_padded, vocabulary)
    X_val = build_input_data(val_sents_padded, vocabulary)
    return X_trn, Y_trn, X_tst, Y_tst, X_val,Y_val, vocabulary, vocabulary_inv

def load_data_2_obj(data_path, max_length=500, vocab_size=50000, split=0):
    with open(os.path.join(data_path), 'rb') as fin:
        # load data
        df = pd.read_json(data_path)

    df.head()
    df['labels'] = df[df.columns[2:]].values.tolist()
    new_df = df[['description', 'solution_matrix', 'file_name']].copy()
    new_df.head()
    #  [train, test, vocab, catgy] =  []
    # split train_val and test
    sss = ShuffleSplit(n_splits=5, test_size=0.3, random_state=52)

    splits = [(train, test) for train, test in sss.split(new_df.description, new_df.solution_matrix)]
    train_val_index, test_index = splits[split]

    splits = [(train, test) for train, test in sss.split(new_df.description.iloc[train_val_index],
                                                         new_df.solution_matrix.iloc[train_val_index])]

    tmp_idx_train, tmp_idx_val = splits[0]
    train_index = train_val_index[tmp_idx_train]

    train_df = new_df.iloc[train_index].reset_index(drop=True)
    test_df = new_df.iloc[test_index].reset_index(drop=True)

    train = []
    test = []
    for index, row in train_df.iterrows():
        # categories = [i for i, x in enumerate(row['solution_matrix']) if x]
        categories = []
        for obj in row['solution_matrix']:
            categories.append([i for i, x in enumerate(obj) if x])
        train.append({'split': 'train', 'text': row['description'], 'Id': row['file_name'], 'catgy': categories,
                      'num_words': len(row['description'])})
    for index, row in test_df.iterrows():
        categories = []
        for obj in row['solution_matrix']:
            categories.append([i for i, x in enumerate(obj) if x])

        test.append(
            {'split': 'test', 'text': row['description'], 'Id': row['file_name'], 'catgy': categories,
             'num_words': len(row['description'])})

    if len(test) == 0:
        test[:5] = train[:5]
    if arr_length>18:
        trn_sents, Y_trn = load_data_and_labels(train)
        tst_sents, Y_tst = load_data_and_labels(test)
    else:
        trn_sents, Y_trn = load_data_and_labels_n(train)
        tst_sents, Y_tst = load_data_and_labels_n(test)


    trn_sents_padded = pad_sentences(trn_sents, max_length=max_length)
    tst_sents_padded = pad_sentences(tst_sents, max_length=max_length)
    print("len:", len(trn_sents_padded), len(tst_sents_padded))
    vocabulary, vocabulary_inv = build_vocab(trn_sents_padded + tst_sents_padded, vocab_size=vocab_size)
    X_trn = build_input_data(trn_sents_padded, vocabulary)
    X_tst = build_input_data(tst_sents_padded, vocabulary)
    return X_trn, Y_trn, X_tst, Y_tst, vocabulary, vocabulary_inv


# def load_data_n(data_path, max_length=500, vocab_size=50000,split=0):
#     # Load and preprocess data
#     with open(os.path.join(data_path), 'rb') as fin:
#         # df = pd.read_json("../MasterThesis/one_to_4_25noise_shuffled order.json")
#         df = pd.read_json(data_path)
#         # df = pd.read_json("../MasterThesis/two_objects.json")
#         # df = pd.read_json("../MasterThesis/edited_files/all_edited.json")
#
#     df.head()
#     df['labels'] = df[df.columns[2:]].values.tolist()
#     new_df = df[['description', 'solution_matrix', 'file_name']].copy()
#     new_df.head()
#
#     sss = ShuffleSplit(n_splits=5, test_size=0.3, random_state=5)
#
#     splits = [(train, test) for train, test in sss.split(new_df.description, new_df.solution_matrix)]
#     train_val_index, test_index = splits[split]
#
#     train_df = new_df.iloc[train_val_index].reset_index(drop=True)
#     test_df = new_df.iloc[test_index].reset_index(drop=True)
#     train =[]
#     test = []
#     for index, row in train_df.iterrows():
#         # categories = [i for i, x in enumerate(row['solution_matrix']) if x]
#         categories = []
#         for obj in row['solution_matrix']:
#             categories.append([i for i, x in enumerate(obj) if x])
#         train.append({'split': 'train', 'text': row['description'], 'Id': row['file_name'], 'catgy': categories,
#                       'num_words': len(row['description'])})
#     for index, row in test_df.iterrows():
#         categories = []
#         for obj in row['solution_matrix']:
#             categories.append([i for i, x in enumerate(obj) if x])
#
#         test.append(
#             {'split': 'test', 'text': row['description'], 'Id': row['file_name'], 'catgy': categories,
#              'num_words': len(row['description'])})
#
#     trn_sents, Y_trn = load_data_and_labels_n(train)
#     print('1.2')
#
#     tst_sents, Y_tst = load_data_and_labels_n(test)
#     print('1.3')
#     trn_sents_padded = pad_sentences(trn_sents, max_length=max_length)
#     tst_sents_padded = pad_sentences(tst_sents, max_length=max_length)
#     print("len:", len(trn_sents_padded), len(tst_sents_padded))
#     vocabulary, vocabulary_inv = build_vocab(trn_sents_padded + tst_sents_padded, vocab_size=vocab_size)
#     X_trn = build_input_data(trn_sents_padded, vocabulary)
#     X_tst = build_input_data(tst_sents_padded, vocabulary)
#     return X_trn, Y_trn, X_tst, Y_tst, vocabulary, vocabulary_inv
#     # return X_trn, Y_trn, vocabulary, vocabulary_inv


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
