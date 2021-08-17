import pandas
import numpy as np

def get_best_lr(results):
    summary = pandas.DataFrame(columns=results.columns, index=results.index)
    for col in results.columns:
        for row in results.index:
            summary.loc[row][col] = np.mean(results.loc[row][col])
    summary = summary.loc[['auc_10','auc_16','cindex','ppvauc10','ppvauc16']]
    return summary.columns[np.argmax(summary.product(axis=0))]
