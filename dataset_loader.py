import os
import pickle 
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

def read_data(data_pickle_path, downsample = True, flatten = True, train_or_test = "train"):
    """
    Args
        data_pickle_path : path to the data pickle 
        downsample       : whether or not to downsample the MNIST dataset
        flatten          : whether or not to flatten the array 
        train_or_test    : either 'train' or 'test' data 
        
    Return 
        (feature array, label array)
    """
    assert train_or_test in ("train", "test")
    
    with open(data_pickle_path,'rb') as f:
         d = pickle.load(f)
           
    # data looks like {"X_train" : np.array, "X_test" : np.array, "Y_train" : np.array, "Y_test" : np.array}
    data = d.get({True : "downsample", False : "raw"}.get(downsample))
    X    = data.get(f"X_{train_or_test}")
    Y    = data.get(f"Y_{train_or_test}")
    
    if flatten:
        X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
        
    return X, Y



def load_credit_dataset(dset_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads the credit dataset
    Args:
        dset_dir (str): Dataset directory
    Returns:
        A tuple containing the features and labels
    """
    if not os.path.isdir(dset_dir):
        raise Exception("Dataset source file not found")

    dset_path = Path(dset_dir) / "creditcard_fraud_detection.csv"

    df = pd.read_csv(dset_path)
    features = df.copy()

    labels = features.pop("Class")
    features = features.drop(["rownames", "Time"], axis=1)

    features = np.array(features)
    labels = np.array(labels)

    return features, labels


def load_uci_dataset(dset_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads the UCI dataset
    Note:
        Amalgamates both test and train splits
    Args:
        dset_dir (str): Dataset directory
    Returns:
        A tuple containing the features and labels
    """
    if not os.path.isdir(dset_dir):
        raise Exception("Dataset source directory not found")

    train_path = Path(dset_dir) / "isolet1+2+3+4.data"
    test_path = Path(dset_dir) / "isolet5.data"

    if not os.path.isfile(train_path):
        raise Exception("Missing UCI training data")

    if not os.path.isfile(test_path):
        raise Exception("Missing UCI test data")

    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    train_features = np.array(train_df.loc[:, :616])
    train_labels = np.array(train_df.loc[:, 617])

    test_features = np.array(test_df.loc[:, :616])
    test_labels = np.array(test_df.loc[:, 617])

    vowels = np.array([1.0, 5.0, 9.0, 15.0, 21.0])

    train_labels = np.isin(train_labels, vowels).astype(np.int32)
    test_labels = np.isin(test_labels, vowels).astype(np.int32)

    features = np.concatenate((train_features, test_features))
    labels = np.concatenate((train_labels, test_labels))

    return features, labels


def load_cervical_cancer_dataset(
    dset_dir: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Loads the cervical cancer dataset
    Args:
        dset_dir (str): Dataset directory
    Returns:
        A tuple containing the features and labels
    """
    if not os.path.isdir(dset_dir):
        raise Exception("Dataset source directory not found")

    dset_path = Path(dset_dir) / "kag_risk_factors_cervical_cancer.csv"

    dset_df = pd.read_csv(dset_path)
    dset_df = dset_df.replace("?", np.nan)

    # There is an exceedingly large number of NaNs in these two columns
    dset_df = dset_df.drop(
        columns=[
            "STDs: Time since first diagnosis",
            "STDs: Time since last diagnosis",
        ]
    )

    # Drop the remaining NaNs
    dset_df_no_nan = dset_df.dropna()

    labels = np.array(dset_df_no_nan.pop("Biopsy").astype(np.float32))
    features = np.array(dset_df_no_nan.astype(np.float32))

    return features, labels


def load_adult_census_dataset(dset_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads the adult census dataset
    Note:
        Amalgamates both test and train splits
    Args:
        dset_dir (str): Dataset directory
    Returns:
        A tuple containing the features and labels
    """
    if not os.path.isdir(dset_dir):
        raise Exception("Dataset source directory not found")

    train_path = Path(dset_dir) / "adult_processed_train.csv"
    test_path = Path(dset_dir) / "adult_processed_test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_labels = np.array(train_df.pop("income"))
    test_labels = np.array(test_df.pop("income"))

    train_features = np.array(train_df)
    test_features = np.array(test_df)

    features = np.concatenate((train_features, test_features))
    labels = np.concatenate((train_labels, test_labels))

    return features, labels


def get_dataset_loader(
    dset_src: str, dset_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the specified dataset loader, according to dset_name
    Args:
        dset_src (str): Dataset source file/directory
    Returns:
        A tuple containing the features and labels
    """
    if dset_name == "credit_fraud":
        return load_credit_dataset(dset_src)
    elif dset_name == "UCI_ISOLET":
        return load_uci_dataset(dset_src)
    elif dset_name == "cervical_cancer":
        return load_cervical_cancer_dataset(dset_src)
    elif dset_name == "adult_census":
        return load_adult_census_dataset(dset_src)
    elif dset_name == "resized_mnist":
        print(read_data(dset_src))
        return read_data(dset_src)

    else:
        raise Exception("Can't find valid dataset loading function")
