# Abstractive Text Summarization for CNN and DailyMail

Abstractive text summarization is the task of generating a headline or a short summary consisting of a few sentences that captures the salient ideas of an article or a passage. We use the adjective ‘abstractive’ to denote a summary that is not a
mere selection of a few existing passages or sentences extracted from the source, but a compressed paraphrasing of the main contents of the document, potentially using vocabulary unseen in the source document.

CNN/Daily Mail is a dataset for text summarization. Human generated abstractive summary bullets were generated from news stories in CNN and Daily Mail websites as questions (with one of the entities hidden), and stories as the corresponding
passages from which the system is expected to answer the fill-in the-blank question. The authors released the scripts that crawl, extract, and generate pairs of passages and questions from these websites.


# HOW TO DOWNLOAD DATA

1. Go to the "Code" folder and execute following statements:

wget https://gwu.box.com/shared/static/b0i9v85is577gaavz0s4ef72bfpza1re.csv

wget https://gwu.box.com/shared/static/2siztdfwbiglakocgm7ugiewgopdgvl5.csv

2. Then Rename files as follows:

mv 2siztdfwbiglakocgm7ugiewgopdgvl5.csv dailymail_stories.csv

mv b0i9v85is577gaavz0s4ef72bfpza1re.csv cnn_stories.csv


# HOW TO RUN:

Simply execute the BART_Transformer.py file in the Code folder








