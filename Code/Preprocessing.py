# Libraries for general purpose
# Text cleaning
# Seed for reproducibility
import os
import random
import re
import string

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
# PyTorch LSTM
import torch
from cleantext import clean
from nltk import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from torch.utils.data import TensorDataset
from transformers import GPT2Tokenizer, AutoTokenizer

OR_PATH = os.getcwd()
os.chdir("..")  # Change to the parent directory
PATH = os.getcwd()
DATA_DIR = os.getcwd() + os.path.sep + 'Code' + os.path.sep
sep = os.path.sep

os.chdir(OR_PATH)  # Come back to the folder where the code resides , all files will be left on this directory

stop_words = set(stopwords.words('english'))


def accuracy_metric(y_true, y_pred):
    res = accuracy_score(y_true, y_pred)
    return res


seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

# ------------------------------------------------------ #
# hyper parameters
# ------------------------------------------------------ #

LR = 0.00001  # Learning rate 3e-4, 5e-5, 3e-5, 2e-5
DROPOUT = 0.5  # LSTM Dropout
BIDIRECTIONAL = False  # Boolean value to choose if to use a bidirectional LSTM or not

MAX_LEN = 256
TRAIN_BATCH_SIZE = 3
VALID_BATCH_SIZE = 3
EPOCHS = 1
LEARNING_RATE = LR  # 1e-05

# ------------------------------------------------------ #
# setting up GPU
# ------------------------------------------------------ #

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# set style for plots
sns.set_style("whitegrid")
sns.despine()
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlepad=10)


# ------------------------------------------------------ #
# Setting up functions to handle data
# ------------------------------------------------------ #

def conf_matrix(y, y_pred, title, labels):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax = sns.heatmap(confusion_matrix(y, y_pred), annot=True, cmap="Purples", fmt='g', cbar=False,
                     annot_kws={"size": 30})
    plt.title(title, fontsize=25)
    ax.xaxis.set_ticklabels(labels, fontsize=16)
    ax.yaxis.set_ticklabels(labels, fontsize=14.5)
    ax.set_ylabel('Test', fontsize=25)
    ax.set_xlabel('Predicted', fontsize=25)
    plt.show()


# ------------------------------------------------------ #
# CUSTOM DEFINED FUNCTIONS TO CLEAN THE text
# ------------------------------------------------------ #

# Clean emojis from text
def strip_emoji(text):
    return clean(text, no_emoji=True)


# Remove punctuations, links, stopwords, mentions and \r\n new line characters

def strip_all_entities(text):
    text = text.replace('\r', '').replace('\n', ' ').lower()  # remove \n and \r and lowercase
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)  # remove links and mentions
    text = re.sub(r'[^\x00-\x7f]', r'', text)  # remove non utf8/ascii characters such as '\x9a\x91\x97\x9a\x97'
    banned_list = string.punctuation
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    text = [word for word in text.split() if word not in stop_words]
    text = ' '.join(text)
    text = ' '.join(word for word in text.split() if len(word) < 14)  # remove words longer than 14 characters
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    return text


# remove contractions

def decontract(text):
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    return text


# clean hashtags at the end of the sentence and keep those in the middle of the sentence by removing just the "#" symbol

def clean_hashtags(tweet):
    new_tweet = " ".join(word.strip() for word in
                         re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', tweet))  # remove last hashtags
    new_tweet2 = " ".join(word.strip() for word in
                          re.split('#|_', new_tweet))  # remove hashtags symbol from words in the middle of the sentence
    return new_tweet2


# Filter special characters such as "&" and "$" present in some words
def filter_chars(a):
    sent = []
    for word in a.split(' '):
        if ('$' in word) | ('&' in word):
            sent.append('')
        else:
            sent.append(word)
    return ' '.join(sent)


# Remove multiple sequential spaces
def remove_mult_spaces(text):
    return re.sub("\s\s+", " ", text)


# Stemming
def stemmer(text):
    tokenized = nltk.word_tokenize(text)
    ps = PorterStemmer()
    return ' '.join([ps.stem(words) for words in tokenized])


# Lemmatization
# NOTE:Stemming seems to work better for this dataset
def lemmatize(text):
    tokenized = nltk.word_tokenize(text)
    lm = WordNetLemmatizer()
    return ' '.join([lm.lemmatize(words) for words in tokenized])


# Then we apply all the defined functions in the following order
def deep_clean(text):
    # text = strip_emoji(text)
    text = decontract(text)
    text = strip_all_entities(text)
    text = clean_hashtags(text)
    text = filter_chars(text)
    text = remove_mult_spaces(text)
    # text = stemmer(text)
    return text


# ------------------------------------------------------------------------------------------------------#
# ----------------------------------- Train - Data Preparation -----------------------------------------#
# ------------------------------------------------------------------------------------------------------#

# Read Data

df_orig = pd.read_csv("dailymail_stories.csv")

df = df_orig[:100000]

#  Cleaning text

for j in ['stories', 'highlights']:
    texts_new = []
    for t in df[j]:
        texts_new.append(deep_clean(t))

    if j == 'stories':
        df['text'] = texts_new
    else:
        df['summary'] = texts_new

print(df.head())

train_df = df[['text', 'summary']]

# ------------------------------------------------------------------------------------------------------#
# ----------------------------------- Test - Data Preparation ------------------------------------------#
# ------------------------------------------------------------------------------------------------------#


# Read Data

df_test_orig = pd.read_csv("cnn_stories.csv")

df_test = df_test_orig[:10000]

#  Cleaning text

for j in ['stories', 'highlights']:
    texts_new = []
    for t in df_test[j]:
        texts_new.append(deep_clean(t))

    if j == 'stories':
        df_test['text'] = texts_new
    else:
        df_test['summary'] = texts_new

print(df_test.head())

test_df = df_test[['text', 'summary']]

# ------------------------------------------------------------------------------------------------------#
# ------------------------------ Creating Validation Dataset--------------------------------------------#
# ------------------------------------------------------------------------------------------------------#

train_size = 0.8
train_df2 = train_df.sample(frac=train_size, random_state=seed_value)
val_df = train_df.drop(train_df2.index).reset_index(drop=True)
train_df = train_df2.reset_index(drop=True)

# ------------------------------------------------------------------------------------------------------#
# ------------------------------ Creating Customer Dataset Loader --------------------------------------#
# ------------------------------------------------------------------------------------------------------#

# tokenizer = AutoTokenizer.from_pretrained('t5-small')
tokenizer = AutoTokenizer.from_pretrained('ainize/bart-base-cnn')

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

pad_on_right = tokenizer.padding_side == "right"


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.df = df
        self.title = df['text']
        self.targets = self.df['summary']
        self.max_len = max_len

    def __len__(self):
        return len(self.title)

    def __getitem__(self, index):
        title = str(self.title[index])
        title = " ".join(title.split())

        targets = str(self.targets[index])
        targets = " ".join(targets.split())

        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        labels = self.tokenizer.encode_plus(
            targets,
            None,
            add_special_tokens=True,
            max_length=90,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            # 'token_type_ids': inputs["token_type_ids"].flatten(),
            # 'targets': torch.FloatTensor(self.targets[index])
            'labels': labels['input_ids'].flatten()
        }


train_dataset = CustomDataset(train_df, tokenizer, MAX_LEN)
valid_dataset = CustomDataset(val_df, tokenizer, MAX_LEN)
test_dataset = CustomDataset(test_df, tokenizer, MAX_LEN)


def create_DataLoaders():
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=TRAIN_BATCH_SIZE,
                                                   shuffle=True,
                                                   num_workers=0
                                                   )

    val_dataloader = torch.utils.data.DataLoader(valid_dataset,
                                                 batch_size=VALID_BATCH_SIZE,
                                                 shuffle=False,
                                                 num_workers=0
                                                 )

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=VALID_BATCH_SIZE,
                                                  shuffle=False,
                                                  num_workers=0
                                                  )

    return train_dataloader, test_dataloader, val_dataloader, tokenizer
