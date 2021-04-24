import string
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from sklearn.preprocessing import LabelEncoder

NUM_CLASS = 11014
MAX_LENGTH = 30


def df_loc_by_list(df, key, values):
    df = df.loc[df[key].isin(values)]
    df = df.assign(sort=pd.Categorical(df[key], categories=values, ordered=True))
    df = df.sort_values('sort')
    # df = df.reset_index()
    df = df.drop('sort', axis=1)
    return df


def removePunctuation(text):
    punc_translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    return text.translate(punc_translator)


class ProductTextDataset(data.Dataset):
    def __init__(self, data_dir, fold, csv, mode, tokenizer):
        self.data_dir = data_dir
        self.fold = fold
        self.csv = csv
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_length = 24
        self.num_classes = NUM_CLASS

        self.df = pd.concat([pd.read_csv(data_dir + '/%s' % f) for f in self.csv])
        self.preprocess()

        samples = list(np.concatenate([np.load(data_dir + '/fold/%s' % f, allow_pickle=True) for f in self.fold]))
        self.df = df_loc_by_list(self.df, 'posting_id', samples)
        self.labels = self.df['label_code'].values

        self.df['title_clean'] = self.df['title'].apply(removePunctuation)
        texts = list(self.df['title_clean'].apply(lambda o: str(o)).values)
        self.encodings = tokenizer(texts,
                                   padding=True,
                                   truncation=True,
                                   max_length=self.max_length)

    def __getitem__(self, index):
        # putting each tensor in front of the corresponding key from the tokenizer
        # HuggingFace tokenizers give you whatever you need to feed to the corresponding model
        item = {key: torch.tensor(values[index]) for key, values in self.encodings.items()}
        # when testing, there are no targets so we won't do the following
        item['labels'] = torch.tensor(self.labels[index]).long()

        return item

    def __str__(self):
        length = len(self)

        string = ''
        string += '\tmode  = %s\n' % self.mode
        string += '\tfold = %s\n' % self.fold
        string += '\tcsv   = %s\n' % str(self.csv)
        string += '\t\tlen  = %5d\n' % length

        return string

    def __len__(self):
        return len(self.df)

    def fold_name(self):
        return self.fold[0].split('_')[1]

    def num_classes(self):
        return self.num_classes

    def preprocess(self):
        lbl_encoder = LabelEncoder()
        self.df['label_code'] = lbl_encoder.fit_transform(self.df['label_group'])

        self.num_classes = self.df['label_code'].nunique()

        # https://www.kaggle.com/moeinshariatnia/indonesian-distilbert-finetuning-with-arcmargin
        # title_lengths = self.df['title'].apply(lambda x: len(x.split(" "))).to_numpy()
        # print(f"MIN words: {title_lengths.min()}, MAX words: {title_lengths.max()}")
        # self.max_length = title_lengths.max()


class ProductTextTestDataset(data.Dataset):
    def __init__(self, data_dir, csv, tokenizer):
        self.data_dir = data_dir
        self.csv = csv
        self.tokenizer = tokenizer
        self.max_length = 48
        self.num_classes = NUM_CLASS

        self.df = pd.concat([pd.read_csv(data_dir + '/%s' % f) for f in self.csv])
        texts = list(self.df['title'].apply(lambda o: str(o)).values)
        self.encodings = tokenizer(texts,
                                   padding=True,
                                   truncation=True,
                                   max_length=self.max_length)

    def __getitem__(self, index):
        # putting each tensor in front of the corresponding key from the tokenizer
        # HuggingFace tokenizers give you whatever you need to feed to the corresponding model
        item = {key: torch.tensor(values[index]) for key, values in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.df)
