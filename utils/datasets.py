import os
import random
from pathlib import Path
from typing import Callable, List, Tuple, Optional, Iterable, Dict, Union
from typing_extensions import TypedDict, Unpack, Required, NotRequired

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
import wandb
from torch.utils.data import Dataset
import torch.nn as nn


def format_mrn(mrn):
    return str(mrn).strip().zfill(20)


class CedarsDatasetTypeAnnotations(TypedDict, total=False):
    """A dummy class used to make IDE autocomplete and tooltips work properly with how we pass **kwargs through in subclasses of CedarsDataset."""
    data_path: Required[Union[Path, str]]
    manifest_path: Required[Union[Path, str]]
    split: NotRequired[str]
    labels: NotRequired[Iterable[str]]
    extra_inputs: NotRequired[Iterable[str]]
    update_manifest_func: NotRequired[Callable[[pd.DataFrame], pd.DataFrame]]
    subsample: NotRequired[Union[Path, str]]
    augmentations: NotRequired[Union[Iterable[Callable[[torch.Tensor], torch.Tensor]], Callable[[dict], dict], nn.Module]]
    apply_augmentations_to: NotRequired[Iterable[str]]
    verify_existing: NotRequired[bool]
    drop_na_labels: NotRequired[bool]
    verbose: NotRequired[bool]


class CedarsDataset(Dataset):
    """
    Generic parent class for several differnet kinds of common datasets we use here at Cedars CVAIR.

    Expects to be used in a scenario where you have a big folder full of input examples (videos, ecgs, 3d arrays, images, etc.) and a big CSV that contains metadata and labels for those examples, called a 'manifest'.

    Args:
        data_path: Path to a directory full of files you want the dataset to load from.
        manifest_path: Path to a CSV or Parquet file containing the names, labels, and/or metadata of your files.
        split: Optional. Allows user to select which split of the manifest to use, assuming the presence of a categorical 'split' column. Defaults to None, meaning that the entire manifest is used by default.
        extra_inputs: Optional. A list of column names in the manifest that contain additional inputs to the model. Defaults to None.
        labels: Optional. Name(s) of column(s) in your manifest which contain training labels, in the order you want them returned. If set to None, the dataset will not return any labels, only filenames and inputs. Defaults to None.
        update_manifest_func: Optional. Allows user to pass in a function to preprocess the manifest after it is loaded, but before the dataset does anything to it.
        subsample: Optional. A number indicating how many examples to randomly subsample from the manifest. Defaults to None.
        verbose: Whether to print out progress statements when initializing. Defaults to True.
        augmentations: Optional. Can be a list of augmentation functions which take in a tensor and return a tensor, a single custom augmentation function which takes in a dict and returns a dict, or a single nn.Module. Defaults to None.
        apply_augmentations_to: Optional. A list of strings indicating which batch elements to apply augmentations to. Defaults to ("primary_input").
    """

    def __init__(
        self,
        data_path,
        manifest_path=None,
        split=None,
        labels=None,
        extra_inputs=None,
        update_manifest_func=None,
        subsample=None,
        augmentations=None,
        apply_augmentations_to=("primary_input",),
        verify_existing=True,
        drop_na_labels=True,
        verbose=True,
    ):

        self.data_path = Path(data_path)
        self.augmentations = augmentations
        self.apply_augmentations_to = apply_augmentations_to
        self.extra_inputs = extra_inputs
        self.labels = labels

        if isinstance(self.augmentations, nn.Module):
            self.augmentations = [self.augmentations]

        if (self.labels is None) and verbose:
            print(
                "No label column names were provided, only filenames and inputs will be returned."
            )
        if (self.labels is not None) and isinstance(self.labels, str):
            self.labels = [self.labels]
        if (self.extra_inputs is not None) and isinstance(self.extra_inputs, str):
            self.extra_inputs = [self.extra_inputs]

        # Read manifest file
        if manifest_path is not None:
            self.manifest_path = Path(manifest_path)
        else:
            self.manifest_path = self.data_path / "manifest.csv"

        if self.manifest_path.exists():
            if self.manifest_path.suffix == ".csv":
                self.manifest = pd.read_csv(self.manifest_path, low_memory=False)
            elif self.manifest_path.suffix == ".parquet":
                self.manifest = pd.read_parquet(self.manifest_path)
        else:
            self.manifest = pd.DataFrame(
                {
                    "filename": os.listdir(self.data_path),
                }
            )

        # do manifest processing that's specific to a given task (different from update_manifest_func,
        # exists as a method overridden in child classes)
        self.manifest = self.process_manifest(self.manifest)

        # Apply user-provided update function to manifest
        if update_manifest_func is not None:
            self.manifest = update_manifest_func(self, self.manifest)

        # Usually set to "train", "val", or "test". If set to None, the entire manifest is used.
        if split is not None:
            self.manifest = self.manifest[self.manifest["split"] == split]
        if verbose:
            print(
                f"Manifest loaded. \nSplit: {split}\nLength: {len(self.manifest):,}"
            )

        # Make sure all files actually exist. This can be disabled for efficiency if
        # you have an especially large dataset
        if verify_existing and "filename" in self.manifest:
            old_len = len(self.manifest)
            existing_files = os.listdir(self.data_path)
            self.manifest = self.manifest[
                self.manifest["filename"].isin(existing_files)
            ]
            new_len = len(self.manifest)
            if verbose:
                print(
                    f"{old_len - new_len} files in the manifest are missing from {self.data_path}."
                )
        elif (not verify_existing) and verbose:
            print(
                f"self.verify_existing is set to False, so it's possible for the manifest to contain filenames which are not present in {data_path}"
            )

        # Option to subsample dataset for doing smaller, faster runs
        if subsample is not None:
            if isinstance(subsample, int):
                self.manifest = self.manifest.sample(n=subsample)
            else:
                self.manifest = self.manifest.sample(frac=subsample)
            if verbose:
                print(f"{subsample} examples subsampled.")

        # Make sure that there are no NAN labels
        if (self.labels is not None) and drop_na_labels:
            old_len = len(self.manifest)
            self.manifest = self.manifest.dropna(subset=self.labels)
            new_len = len(self.manifest)
            if verbose:
                print(
                    f"{old_len - new_len} examples contained NaN value(s) in their labels and were dropped."
                )
        elif (self.labels is not None) and (not drop_na_labels):
            print(
                "drop_na_labels is set to False, so it's possible for the manifest to contain NaN-valued labels."
            )

        # Save manifest to weights and biases run directory
        if wandb.run is not None:
            run_data_path = Path(wandb.run.dir).parent / "data"
            if not run_data_path.is_dir():
                run_data_path.mkdir()

            save_name = "manifest.csv"
            if split is not None:
                save_name = f"{split}_{save_name}"

            self.manifest.to_csv(run_data_path / save_name)

            if verbose:
                print(f"Copy of manifest saved to {run_data_path}")

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        output = {}
        row = self.manifest.iloc[index]
        if "filename" in row:
            output["filename"] = row["filename"]
        if self.labels is not None:
            output["labels"] = torch.FloatTensor(row[self.labels])
        file_results = self.read_file(self.data_path / output["filename"], row)
        if isinstance(file_results, dict):
            output.update(file_results)
        else:
            output["primary_input"] = file_results

        if self.extra_inputs is not None:
            output["extra_inputs"] = row["extra_inputs"]

        if self.augmentations is not None:
            output = self.augment(output)

        return output

    def process_manifest(self, manifest: pd.DataFrame) -> pd.DataFrame:
        if "mrn" in manifest.columns:
            manifest["mrn"] = manifest["mrn"].apply(format_mrn)
        if "study_date" in manifest.columns:
            manifest["study_date"] = pd.to_datetime(manifest["study_date"])
        if "dob" in manifest.columns:
            manifest["dob"] = pd.to_datetime(
                manifest["dob"], infer_datetime_format=True, errors="coerce"
            )
        if ("study_date" in manifest.columns) and ("dob" in manifest.columns):
            manifest["study_age"] = (
                manifest["study_date"] - manifest["dob"]
            ) / np.timedelta64(1, "Y")
        return manifest

    def augment(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        if isinstance(self.augmentations, Iterable):
            # would use torch.stack here for cleanliness, but it seems that torchvision
            # transforms v1's claims about supporting "arbitrary leading dimensions" is
            # hogwash. they only support up to 4D. so we have to concatenate along the
            # channel dimension, then apply the augmentations, then split along the channel
            # dimension.
            augmentable_inputs = torch.cat(
                [output_dict[key] for key in self.apply_augmentations_to], dim=0
            )  # (C*N, T, H, W)

            for aug in self.augmentations:
                augmentable_inputs = aug(augmentable_inputs)

            place = 0
            for i, key in enumerate(self.apply_augmentations_to):
                n_channels = output_dict[key].shape[0]
                output_dict[key] = augmentable_inputs[place:place+n_channels]
                place += n_channels

        elif isinstance(self.augmentations, Callable):
            output_dict = self.augmentations(output_dict)

        else:
            raise Exception(
                "self.augmentations must be either an Iterable of augmentations or a single custom augmentation function."
            )

        return output_dict

    def read_file(self, filepath: Path, row: Optional[pd.Series] = None) -> torch.Tensor:
        raise NotImplementedError


class ECGSingleLeadDataset(CedarsDataset):
    def __init__(
        self,
        # CedarsDataset params
        data_path: Union[Path, str],
        manifest_path: Union[Path, str] = None,
        labels: Union[List[str], str] = None,
        update_manifest_func: Callable = None,
        subsample: float = None,
        verbose: bool = True,
        verify_existing: bool = True,
        drop_na_labels: bool = True,
        **kwargs,
    ):
        """
        Args:
            leads: List[str] -- which leads you want passed to the model. Defaults to all 12.
        """

        super().__init__(
            data_path=data_path,
            manifest_path=manifest_path,
            labels=labels,
            update_manifest_func=update_manifest_func,
            subsample=subsample,
            verbose=verbose,
            verify_existing=verify_existing,
            drop_na_labels=drop_na_labels,
            **kwargs,
        )


    def read_file(self, filepath, row=None):
        # ECGs are usually stored as .npy files.
        try:
            file = np.load(filepath)
        except Exception as e:
            print(filepath)
            print(e)

        file = torch.tensor(file).float().unsqueeze(0)


        # Final shape should ideally be NumLeadsxTime(or NumLeadsxTime depending on the resolution of the ECG)
        return file



