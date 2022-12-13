import torch
#from torchtext.legacy import data
#from torchtext.legacy import datasets
import torch.nn as nn
import torch.optim as optim
import time
import pandas as pd
from html.parser import HTMLParser
from lxml import html

from bs4 import BeautifulSoup
import os
import glob
import re
from sklearn.model_selection import train_test_split
#%%
folderpath = os.getcwd()

#%%
#os.chdir(folderpath+f'/FinalProject')

#%%
#os.chdir(f'/home/ubuntu/NLP/FinalProject/')

#%%
print(folderpath)

#%%

#%%
'''
# Checking the Raw HTML
# CNN
filepath = folderpath+f'/Data/CNN_DailyMail/CNN/cnn/downloads/'
with open(filepath+r'7d4fca888f5195971a9bfa0abeaf12c84cb537aa.html',"r") as f:
    raw_html = f.read()

print(raw_html)

#%%
# Checking using Beautiful Soup
# Beautiful Soup
with open(filepath+r'7d4fca888f5195971a9bfa0abeaf12c84cb537aa.html',"r") as f:
    doc = BeautifulSoup(f, 'html.parser')

#%%
# Checking Story
filepath = folderpath+f'/Data/CNN_DailyMail/CNN/cnn/stories/'
with open(filepath+r'7f33075eec4a1398ea838d197d8f5f3f1d6a90d9.story',"r") as f:
    raw_story = f.read()
print(raw_story)

#%%
# Checking Questions
filepath = folderpath+f'/Data/CNN_DailyMail/CNN/cnn/questions/test/'
with open(filepath+r'7dfbd10ae7deb83eec45f35b8ce3cf423980363f.question',"r") as f:
    raw_questions = f.read()
print(raw_questions)


#%%
# ==========================
# Loading all the stories #
# ==========================
filepath = folderpath+f'/Data/CNN_DailyMail/CNN/cnn/stories/'
print(filepath)
#%%
def load_stories(filepath):
    stories = []
    for filename in os.listdir(filepath):
        if filename.endswith('.story'):
            with open(filepath + filename, "r") as f:
                raw_text = f.read()
            stories.append(raw_text)


    df = pd.DataFrame(stories, columns=['stories'])

    return df

#%%
# Creating DataFrame for CNN Stories.
cnn_stories = load_stories(filepath)


#%%
# Splitting Highlights from stories
def split_highlights(df):
    highlights = df['stories'].str.split('@highlight', n=1, expand=True)
    df['stories'] = highlights[0]
    df['highlights'] = highlights[1]

    return df

#%%
cnn_stories = split_highlights(cnn_stories)

#%%
cnn_stories.head()

#%%
print(cnn_stories.iloc[0,0])
print(cnn_stories.head())
print(len(cnn_stories))

#%%
# Creating a CSV File
cnn_stories.to_csv('cnn_stories.csv', index=False, encoding = 'utf-8')

#%%
# Loading from CSV
cnn = pd.read_csv('cnn_stories.csv')
#%%
cnn.head()
print(cnn.iloc[0,0])
#%%
# ===============
# DailyMail #####
# ===============

filepath = folderpath+f'/Data/CNN_DailyMail/DailyMail/dailymail/stories/'
print(filepath)
#%%
# ===========================
# Loading Daily Mail Stories
# ===========================

dailymail_stories = load_stories(filepath)

#%%
# Splitting Highlights from Stories:
dailymail_stories = split_highlights(dailymail_stories)

#%%
print(dailymail_stories.iloc[0,0])
#%%
print(dailymail_stories.head())
print(len(dailymail_stories))
#%%
# Converting DailyMail Stories to CSV File:
dailymail_stories.to_csv('dailymail_stories.csv', index=False, encoding = 'utf-8')

'''
#%%
# ==================================================
# ================= Begin Here =====================
# ==================================================
#%%
#os.chdir(folderpath+f'/FinalProject/')

#%%
# Loading from CSV
# Using DailyMail for Training and Validation
dailymail = pd.read_csv('dailymail_stories.csv')

# Using CNN for Test Data
cnn = pd.read_csv('cnn_stories.csv')


#%%
# Taking only sample of the data because of its size:

#daily_sample = dailymail.sample(frac=0.10, random_state=1)
#cnn_sample = cnn.sample(frac=0.10, random_state=1)

daily_sample = dailymail[:100]
cnn_sample = cnn[:50]

#%%
print(daily_sample.iloc[1,0]) # Checking Stories

#%%
# CNN Stories

# Preprocessing
def normalise_text (text):
    text = text.str.lower() # lowercase
    text = text.str.replace(r"\#","") # replaces hashtags
    text = text.str.replace(r"http\S+","URL")  # remove URL addresses
    text = text.str.replace(r"@","")
    text = text.str.replace(r"[^A-Za-z0-9()!?\'\`\"]", " ")
    text = text.str.replace("\s{2,}", " ")
    return text

#%%
'''
def normalise_summary (text):
    #text = text.str.lower() # lowercase
    #text = text.str.replace(r"\#","") # replaces hashtags
    #text = text.str.replace(r"http\S+","URL")  # remove URL addresses
    #text = text.str.replace(r"@highlight","")
    text = text.str.replace(r"[^A-Za-z0-9()!?\'\`\"]", " ")
    text = text.str.replace(r"/^\s+|\s+$/g"," ")
    text = text.str.replace("\s{2,}$", ". ")
    #text = text.str.replace("[^\.]", "")

    return text
'''
#%%
daily_sample["stories"]=normalise_text(daily_sample["stories"])


#%%
#print(daily_sample.iloc[1,0]) # Checking Stories

#%%
#print(dailymail.iloc[0,1]) # Checking Highlights before cleaning

print(daily_sample.head())

#%%
#train["highlights"]=normalise_summary(train["highlights"])


#%%
print(len(daily_sample))

print(len(cnn_sample))

#%%
# Splitting the train data into train and validation

train = daily_sample[:int(round((len(daily_sample)*0.75), 0))] # 0.75 as Train
valid = daily_sample[int(round((len(daily_sample)*0.75), 0)):] # 0.25 as Validation

test = cnn_sample.copy()

#%%

print(train.shape, valid.shape, test.shape)


#%%
from transformers import AutoTokenizer


def model_input():
    model_input = input('Enter which model to run: "t5-small" or "ainize/bart-base-cnn" ?:\n')

    if model_input == 't5-small':
        model_checkpoint = 't5-small'
    elif model_input == 'ainize/bart-base-cnn':
        model_checkpoint = 'ainize/bart-base-cnn'

    return model_checkpoint

model_checkpoint = model_input()

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

pad_on_right = tokenizer.padding_side == "right"
#%%
prefix = "summarize: "
max_input_length = 512
max_target_length = 128

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["stories"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True,padding='max_length')

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["highlights"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

#%%

from datasets import Dataset

train = Dataset.from_pandas(train)
valid = Dataset.from_pandas(valid)

#%%

tokenized_train = train.map(preprocess_function, batched=True)
tokenized_valid = valid.map(preprocess_function, batched=True)

#%%
import transformers
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

#%%
# ======================= FINE TUNING ============================

from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

#%%
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

#%%

batch_size = 5
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

#%%

# ==================== EVALUATION =================================

#import evaluate

#rouge = evaluate.load("rouge")



#%%

import nltk
import numpy as np
from datasets import load_metric

metric = load_metric("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


#%%

import gc
gc.collect()

#%%

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


print("[INFO] training using {}".format(torch.cuda.get_device_name(0)))

#%%

torch.cuda.empty_cache()

#%%
os.environ["WANDB_DISABLED"] = "true"

#%%
#model_name = 'Group3'
model_name = model_checkpoint.split("/")[-1]
training_args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=True
)
#%%
#training_args = Seq2SeqTrainingArguments(
#    f"{model_name}-finetuned-newsarticles",
#    evaluation_strategy = "steps",
#    learning_rate=2e-5,
#    per_device_train_batch_size=batch_size,
#    per_device_eval_batch_size=batch_size,
#    gradient_accumulation_steps=2,
#    weight_decay=0.01,
#    save_steps=1000,
#    save_strategy='steps',
#    eval_steps=1000,
#    logging_steps=1000,
#    save_total_limit=3,
#    num_train_epochs=1,
#    load_best_model_at_end = True,
#    predict_with_generate=True,
#    fp16=True
#)


#%%

trainer = Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

#%%

trainer.train()

#%%
eval_dataset = Dataset.from_pandas(test)

#%%
eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True)

#%%
predict_results = trainer.predict(
            eval_dataset,max_length=128, num_beams=3)

#%%
metrics = predict_results.metrics

#%%

print(metrics)

#%%


if training_args.predict_with_generate:
    predictions = tokenizer.batch_decode(predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    predictions = [pred.strip() for pred in predictions]

#%%


print("Predicted Summary:\n\n", predictions[:1])


#%%

original_text = test['stories'].to_list()

#%%
original_summary = test['highlights'].to_list()

#%%
print("\nOriginal Text:\n\n", original_text[:1])
print("\nOriginal Summary: \n\n", original_summary[:1])



#%%