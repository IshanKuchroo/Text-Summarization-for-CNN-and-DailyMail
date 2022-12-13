# Libraries for general purpose
# Text cleaning
import datetime
# Seed for reproducibility
import time

import accelerator as accelerator
import evaluate
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
# PyTorch LSTM
import torch
from accelerate import Accelerator
from tqdm import tqdm
# Transformers library for BERT
from transformers import AutoModelForSeq2SeqLM, get_scheduler

from Preprocessing import create_DataLoaders
from SHAP_Function import shap_explanation

# Data preprocessing
# Naive Bayes
# from utils import add_special_tokens, beam_search, generate_beam_sample, generate_sample, sample_seq, set_seed, \
#     top_k_top_p_filtering

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

# ------------------------------------------------------ #
# setting up GPU
# ------------------------------------------------------ #

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# ------------------------------------------------------------------------------------------------------#
# -----------------------------------MODEL BUILDING ----------------------------------------------------#
# ------------------------------------------------------------------------------------------------------#

train_dataloader, val_dataloader, test_dataloader, tokenizer, test_df = create_DataLoaders()


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # ROUGE expects a newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


model = AutoModelForSeq2SeqLM.from_pretrained("ainize/bart-base-cnn")

model.resize_token_embeddings(len(tokenizer))

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

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
            b_input_ids = batch['input_ids'].to(device, dtype=torch.long)
            b_attn_mask = batch['attention_mask'].to(device, dtype=torch.long)
            # token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
            # b_labels = batch['targets'].to(device, dtype=torch.float)
            # outputs = model(b_input_ids, b_attn_mask)

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

            b_input_ids = batch['input_ids'].to(device, dtype=torch.long)
            b_attn_mask = batch['attention_mask'].to(device, dtype=torch.long)
            # token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
            # b_labels = batch['targets'].to(device, dtype=torch.float)
            # outputs = model(b_input_ids, b_attn_mask)

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
    plt.rcParams["figure.figsize"] = (16, 8)

    # Plot the learning curve.
    plt.plot(df_stats['Training Loss'], marker='o', color='blue', label="Training")
    plt.plot(df_stats['Valid. Loss'], marker='o', color='red', label="Validation")
    # plt.plot(df_stats['ROUGEL'], marker='*', linestyle='dashed', color='green', label="ROUGE")

    # Label the plot.
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([1, 2, 3, 4])

    plt.show()

    plt.rcParams["figure.figsize"] = (16, 8)

    # Plot the learning curve.
    plt.plot(df_stats['ROUGE1'], marker='*', color='blue', label="ROUGE1")
    plt.plot(df_stats['ROUGE2'], marker='*', color='red', label="ROUGE2")
    plt.plot(df_stats['ROUGEL'], marker='o', color='red', label="ROUGEL")
    plt.plot(df_stats['ROUGELSUM'], marker='o', color='red', label="ROUGELSUM")

    # Label the plot.
    plt.title("ROUGE Score")
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

print('The BART model has {:} different named parameters.\n'.format(len(params)))

print('***** Embedding Layer *****\n')

for p in params[0:2]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n***** First Transformer *****\n')

for p in params[2:14]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n***** Output Layer *****\n')

for p in params[-2:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

trained_model, test_dataloader = accelerator.prepare(trained_model, test_dataloader)

preds, labels = evaluate(test_dataloader, trained_model)

print("Predicted Summary: ", preds)
print("\n")
print("Original Summary: ", labels)

shap_explanation(test_df, trained_model, tokenizer)


