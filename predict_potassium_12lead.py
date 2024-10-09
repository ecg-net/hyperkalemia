import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from utils.datasets import ECGDataset
from utils.models import EffNet
from utils.training_models import RegressionModel


# +
# This is the path where your data samples are stored.
data_path = "your/ecg/data/folder"

# This is the path where your manifest, containing filenames for inference to be run on, is stored.
manifest_path = 'your/manifest/path'
# -


# Initialize a dataset that contains the examples you want to run prediction on. 
test_ds = ECGDataset(
    split="test",
    data_path=data_path,
    manifest_path=manifest_path,
    update_manifest_func=None,
)

# Wrap the dataset in a dataloader to handle batching and multithreading.
test_dl = DataLoader(
    test_ds, 
    num_workers=16, 
    batch_size=256, 
    drop_last=False, 
    shuffle=False
)

# Initialize the "backbone", the core model weights that will act on the data.
backbone = EffNet(input_channels=12, output_neurons=1)

model = RegressionModel(backbone)

weights = torch.load("model_12_lead.pt")
print(model.load_state_dict(weights))

# +
trainer = Trainer(accelerator="gpu", devices=1)

trainer.predict(model, dataloaders=test_dl)
