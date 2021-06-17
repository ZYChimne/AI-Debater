import pandas as pd
from sklearn.metrics import classification_report
import csv
import numpy as np

eval_df = pd.read_csv('data/test.txt', sep='\t', header=None, quoting=csv.QUOTE_NONE)
eval_df.columns = ['text_a', 'text_b', 'labels']

preds = []
with open('outputs/submission.csv') as f:
# with open('sample_submission/submission.csv') as f:
    for line in f:
        preds.append(line.strip())

acc = np.sum(np.array(preds)==eval_df['labels'])*1.0/len(eval_df['labels'])
print (f'Submission Accuracy: {acc:.4f}')

