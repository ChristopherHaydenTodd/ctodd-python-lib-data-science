#!/usr/bin/env python3
"""
    Library for helping store/load/persist data science
    models using Python libraries
"""

# Python Library Imports
import sys
import os
import logging
import pandas as pd
import sklearn
import pickle
from sklearn.model_selection import train_test_split

###
# Test/Train Split
###


def store_model_as_pickle(filename, config={}, metadata={}):
    """
    Purpose:
        Store a model in memory to a .pkl file for later
        usage. ALso store a .config file and .metadata
        file with information about the model
    Args:
        filename (String): Filename of a pickled model (.pkl)
        config (Dict): Configuration data for the model
        metadata (Dict): Metadata related to the model/training/etc
    Return:
        N/A
    """

    print("No-op")


def load_pickled_model(filename):
    """
    Purpose:
        Load a model that has been pickled and stored to
        persistance storage into memory
    Args:
        filename (String): Filename of a pickled model (.pkl)
    Return:
        model (Pickeled Object): Pickled model loaded from .pkl
    """

    if not os.path.isfile(filename):
        error_msg = f"Model Filename ({filename}) does not exist, exiting"
        logging.error(error_msg)
        raise Exception(error_msg)

    try:
        with open(filename, 'rb') as model_file:
            model = pickle.load(model_file)
    except Exception as err:
        logging.exception(f"Exception Loading Pickle from File into Memory: {err}")
        raise err

    return model
