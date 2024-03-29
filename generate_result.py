# An example of how to run inference using a model trained with cvair.
# We want to use a pretrained model to make predictions on a dataset of new examples.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def analyze_predictions(filename, noise_threshold=0.5):
    df = pd.read_csv(filename)
    df.dropna(subset=['label'], inplace=True)
    
    #get rid of noised ecg, adjust threshold based on your ecg data
    df = df[df['noise_preds_normal'] < noise_threshold]  
    
    #set target columns
    df['K_6.5'] = np.where(df['label'] > 6.5, 1, 0)
    df['K_3'] = np.where(df['label'] < 3.0, 1, 0)
    
    #average prediction result
    df['ecg_filename'] = df['filename'].str.split('_').str[0] + '_' + df['filename'].str.split('_').str[1] + '_' + df['filename'].str.split('_').str[2]
    mean_prediction = df.groupby('ecg_filename')['preds'].mean().reset_index()
    mean_prediction = mean_prediction.rename(columns={'preds': 'mean_prediction'})
    df = pd.merge(df, mean_prediction, on='ecg_filename', how='left')
    df_single_test = df.drop_duplicates(subset=['ecg_filename'])
    
    #calculate auc, mae
    auc = roc_auc_score(df_single_test['K_6.5'], df_single_test['mean_prediction'])
    mae = (df_single_test['label'] - df_single_test['mean_prediction']).abs().mean()
    
    #save the prediction result
    df_single_test.to_csv('average_prediction.csv')
    
    #plot
    coeffs = np.polyfit(df_single_test['label'], df_single_test['mean_prediction'], 1)
    x = np.linspace(df_single_test['label'].min(), df_single_test['label'].max(), 100000)
    y = np.polyval(coeffs, x)
    plt.scatter(df_single_test['label'], df_single_test['mean_prediction'], label=f'MAE:{mae:.3f}')
    plt.plot(x, y, color='red')
    plt.legend(loc='lower right')
    plt.xlim(2, 10)
    plt.ylim(2, 10)
    plt.savefig('scatter_plot.png')
    
    return auc, mae


if __name__ == "__main__":
    
    filename='dataloader_0_predictions.csv'
    
    auc, mae = analyze_predictions(filename)
    
    print(f'MAE: {mae:.3f}, AUC: {auc:.3f}')
