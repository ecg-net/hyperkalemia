# An example of how to run inference using a model trained with cvair.
# We want to use a pretrained model to make predictions on a dataset of new examples.

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from utils.datasets import ECGSingleLeadDataset
from utils.models import EffNet
from utils.training_models import BinaryClassificationModel


def append_noise_predictions_to_manifest(data_path, manifest_path, weights_path):
    # Initialize a dataset
    test_ds = ECGSingleLeadDataset(
        data_path=data_path,
        manifest_path=manifest_path,
        update_manifest_func=None,
    )

    # Wrap the dataset in a dataloader
    test_dl = DataLoader(
        test_ds, num_workers=16, batch_size=512, drop_last=False, shuffle=False
    )

    # Initialize the backbone model
    backbone = EffNet(input_channels=1, output_neurons=1)

    # Pass the backbone to a wrapper
    model = BinaryClassificationModel(backbone)

    # Load the pretrained weights
    weights = torch.load(weights_path)
    model.load_state_dict(weights)

    # Initialize a Trainer object
    trainer = Trainer(accelerator="gpu", devices=1)

    # Run inference
    trainer.predict(model, dataloaders=test_dl)

    # Read the predictions CSV file
    df = pd.read_csv('dataloader_0_predictions.csv')

    # Normalize predictions
    max_preds = df['preds'].max()
    min_preds = df['preds'].min()
    df['noise_preds_normal'] = (df['preds'] - min_preds) / (max_preds - min_preds)

    # Drop the original predictions column
    df.drop(columns='preds', inplace=True)

    # Save the modified dataframe to the original manifest path
    df.to_csv(manifest_path)



if __name__ == "__main__":

    data_path="/workspace/data/drives/sdc/ecg_database/apple_watch_study/5seconds_250samples_MW_array_applewatch/"

    manifest_path="/workspace/imin/applewatch_potassium/manifest_apple_ecg_5seconds_250samples_MV.csv"
    
    weights_path = "model_best_auc_noise_classifier.pt"
    
    append_noise_predictions_to_manifest(data_path, manifest_path, weights_path)

