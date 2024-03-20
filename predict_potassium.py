# An example of how to run inference using a model trained with cvair.
# We want to use a pretrained model to make predictions on a dataset of new examples.

import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from datasets import ECGSingleLeadDataset
from models import EffNet
from training_models import RegressionModel

# This is the path where your data samples are stored.
data_path = "your/ecg/data/folder"
data_path = "/workspace/data/drives/sdc/ecg_database/apple_watch_study/5seconds_leadI_array_from_cedars_esrd/"

# This is the path where your manifest, containing filenames for inference
# to be run on, is stored.
manifest_path = 'your/manifest/path'
manifest_path="/workspace/imin/applewatch_potassium/5seconds_cedars_leadI_esrd.csv"

# Initialize a dataset that contains the examples you want to run
# prediction on. We want to inference on the test set, so we copy our
# val_ds initialization from our train.py, but change every instace of
# 'val' to 'test'.
test_ds = ECGSingleLeadDataset(
    data_path=data_path,
    manifest_path=manifest_path,
    update_manifest_func=None,
)

# Wrap the dataset in a dataloader to handle batching and multithreading.
test_dl = DataLoader(
    test_ds, num_workers=16, batch_size=512, drop_last=False, shuffle=False
)

# +
# Initialize the "backbone", the core model weights that will act on the data.
backbone = EffNet()

model = RegressionModel(backbone)
# -

# We need load the pretrained weights from a file, then initialize the model
# with them using load_state_dict. If all goes well, the message will say:
# <All keys matched successfully>
weights = torch.load("model_best_mae_5seconds_length.pt")
print(model.load_state_dict(weights))

# Initialize a pl.Trainer object (identical to the one in train.py), and
# call trainer.predict() to run inference.

# Unfortunately, you can't use more than 1 GPU to run inference.
trainer = Trainer(accelerator="gpu", devices=1)

# A file named "predictions.csv" will be saved to the Weights & Biases run
# directory, or if there's no Weights & Biases run active, to the directory
# the script was run in.
trainer.predict(model, dataloaders=test_dl)




