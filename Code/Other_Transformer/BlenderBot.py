# Libraries for general purpose
# Text cleaning
import re
import string
import datetime

import accelerator as accelerator
import evaluate
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from accelerate import Accelerator
from cleantext import clean
from nltk import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from torch.utils.data import TensorDataset
import torch.nn as nn
from tqdm import tqdm
from transformers import pipeline, GPT2Model, GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, BertTokenizerFast, \
    EncoderDecoderModel
# Seed for reproducibility
import random
import time

# Data preprocessing

# Naive Bayes

# PyTorch LSTM
import torch

# Transformers library for BERT
from transformers import AutoModelForSeq2SeqLM, get_scheduler, AutoTokenizer
from transformers import AdamW

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
# from utils import add_special_tokens, beam_search, generate_beam_sample, generate_sample, sample_seq, set_seed, \
#     top_k_top_p_filtering

torch.cuda.empty_cache()

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
# BIDIRECTIONAL = False  # Boolean value to choose if to use a bidirectional LSTM or not

MAX_LEN = 128
# MAX_TARGET_LEN = 128
TRAIN_BATCH_SIZE = 3
VALID_BATCH_SIZE = 3
# EPOCHS = 1
# LEARNING_RATE = LR  # 1e-05
#
# EMBEDDING_DIM = 200
# BATCH_SIZE = 32
# HIDDEN_DIM = 100  # number of neurons of the internal state (internal neural network in the LSTM)
# LSTM_LAYERS = 1  # Number of stacked LSTM layers
#
# # Specify hidden size of BERT, hidden size of the classifier, and number of labels
# n_input = 768
# n_hidden = HIDDEN_DIM  # 50
# n_filters = 100
# n_kernel_size = 1

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

df = df_orig[:2500]

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

df_test = df_test_orig[:500]

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

tokenizer = AutoTokenizer.from_pretrained('facebook/blenderbot_small-90M')
# tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

# if tokenizer.pad_token is None:
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#
# pad_on_right = tokenizer.padding_side == "right"


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
            max_length=64,
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


# ------------------------------------------------------------------------------------------------------#
# -----------------------------------MODEL BUILDING ----------------------------------------------------#
# ------------------------------------------------------------------------------------------------------#

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # ROUGE expects a newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

model = AutoModelForSeq2SeqLM.from_pretrained('facebook/blenderbot_small-90M')
# configuration = GPT2Config.from_pretrained('distilgpt2', output_hidden_states=False)
#
# model = GPT2LMHeadModel.from_pretrained('distilgpt2', config=configuration)

model.resize_token_embeddings(len(tokenizer))

optimizer = AdamW(model.parameters(), lr=LR)
# optimizer = AdamW(model.parameters(), lr=2e-5)

num_train_epochs = 5
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

accelerator = Accelerator()

model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(model, optimizer, train_dataloader,
                                                                         val_dataloader)

rouge_score = evaluate.load("rouge")

progress_bar = tqdm(range(num_training_steps))


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


def train_and_val(train_dataloader, val_dataloader):
    training_stats = []

    for epoch in range(num_train_epochs):

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_train_loss, batch_loss, batch_counts, total_val_loss = 0, 0, 0, 0

        # Training
        model.train()
        for step, batch in enumerate(train_dataloader):

            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            batch_loss = loss.item()
            total_train_loss += batch_loss

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)

        avg_train_loss = total_train_loss / len(train_dataloader)

        training_time = format_time(time.time() - t0_epoch)

        model.eval()

        for step, batch in enumerate(val_dataloader):

            t0_epoch, t0_batch = time.time(), time.time()

            with torch.no_grad():

                outputs = model(**batch)
                loss = outputs.loss

                batch_loss = loss.item()
                total_val_loss += batch_loss

                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )

                generated_tokens = accelerator.pad_across_processes(generated_tokens,
                                                                    dim=1,
                                                                    pad_index=tokenizer.pad_token_id
                                                                    )

                labels = batch["labels"]

                # If we did not pad to max length, we need to pad the labels too
                labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()

                # Replace -100 in the labels as we can't decode them
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]

                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

                rouge_score.add_batch(predictions=decoded_preds, references=decoded_labels)

        avg_val_loss = total_val_loss / len(val_dataloader)

        # Measure how long this epoch took.
        val_time = format_time(time.time() - t0_epoch)

        torch.save(model.state_dict(), './state_dict.pt')

        # Compute metrics
        result = rouge_score.compute()
        # Extract the median ROUGE scores
        # result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        result = {key: value * 100 for key, value in result.items()}
        result = {k: round(v, 4) for k, v in result.items()}
        print(f"Epoch {epoch}:", result)

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'ROUGE1': result['rouge1'],
                'ROUGE2': result['rouge2'],
                'ROUGEL': result['rougeL'],
                'ROUGELSUM': result['rougeLsum'],
                'Training Time': training_time,
                'Validation Time': val_time
            }
        )

    # Create a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data=training_stats)

    # Use the 'epoch' as the row index.
    df_stats = df_stats.set_index('epoch')

    # A hack to force the column headers to wrap.
    # df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])

    # Display the table.
    print(df_stats)

    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 8)

    # Plot the learning curve.
    plt.plot(df_stats['Training Loss'], marker='o', color='blue', label="Training")
    plt.plot(df_stats['Valid. Loss'], marker='o', color='red', label="Validation")
    plt.plot(df_stats['ROUGEL'], marker='*', linestyle='dashed', color='green', label="ROUGE")

    # Label the plot.
    plt.title("Training & Validation Loss + ROUGE Score")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([1, 2, 3, 4])

    plt.show()

    print("Training complete!")

    return model


def evaluate(test_dataloader, model):
    # model.load_state_dict(torch.load('model_{}.pt'.format(NICKNAME), map_location=device))

    for epoch in range(1):

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        for step, batch in enumerate(test_dataloader):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )

                generated_tokens = accelerator.pad_across_processes(generated_tokens,
                                                                    dim=1,
                                                                    pad_index=tokenizer.pad_token_id
                                                                    )

                labels = batch["labels"]

                # If we did not pad to max length, we need to pad the labels too
                labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()

                # Replace -100 in the labels as we can't decode them
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]

                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

                rouge_score.add_batch(predictions=decoded_preds, references=decoded_labels)

        # Compute metrics
        result = rouge_score.compute()
        # Extract the median ROUGE scores
        # result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        result = {key: value * 100 for key, value in result.items()}
        result = {k: round(v, 4) for k, v in result.items()}
        print(f"Epoch {epoch}:", result)

    print("Prediction complete!")

    return decoded_preds, decoded_labels


trained_model = train_and_val(train_dataloader, val_dataloader)

# Get all the model's parameters as a list of tuples.
params = list(trained_model.named_parameters())

print('The GPT-2 model has {:} different named parameters.\n'.format(len(params)))

print('==== Embedding Layer ====\n')

for p in params[0:2]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')

for p in params[2:14]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== Output Layer ====\n')

for p in params[-2:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

trained_model, test_dataloader = accelerator.prepare(trained_model, test_dataloader)

preds, labels = evaluate(test_dataloader, trained_model)
print(preds)
print(labels)

