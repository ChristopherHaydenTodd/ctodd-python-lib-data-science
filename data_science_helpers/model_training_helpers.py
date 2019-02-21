#!/usr/bin/env python3
"""
    Library for helping train data science
    models using Python libraries
"""

# Python Library Imports
import sys
import os
import logging
import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split

###
# Test/Train Split
###

def split_dataframe_for_model_training(
    df, dependent_variable, independent_variables=None, train_size=.70):
    """
        Purpose:
            Takes in DataFrame and creates 4 DataFrames.
            2 DataFrames holding X varib DataFrames and 2 Model Y DataFrames.
            Train size is defaulted at 70% and the split defaults to using
            all passed in columns.
        Args:
            df (Pandas DataFrame): DataFrame to split
            dependent_variable (string): dependent variable being
                that the model is being created to predict
            independent_variables (List of strings): independent variables that
                will be used to predict the dependent varilable. If no columns
                are passed, use all columns in the dataframe except the
                dependent variable.
            train_size (float): Percentage of rows in DataFrame
                to use testing model. Inverse precentage will/can
                be used to test the model's effectiveness
        Return
            train_x (Pandas DataFrame): DataFrame with all independent variables
                for training the model. Size is equal to a percentage of the
                base dataset multiplied by the train size
            test_x (Pandas DataFrame): DataFrame with all independent variables
                for testing the trained model. Size is equal to a percentage
                of the base dataset subtracted by the train size
            train_y_observed (Pandas DataFrame): DataFrame with all dependant
                variables for training the model. Size is equal to a percentage
                of the base dataset multiplied by the train size
            test_y_observed (Pandas DataFrame): DataFrame with all dependant
                variables testing the trained model. Size is equal to a
                percentage of the base dataset multiplied by the train size
    """
    logging.info('Creating Train/Test Split for DataFrame')
    logging.info(
        'Data Train-Size Set to {train_size}. {test_size} for Testing'.format(
            train_size=train_size, test_size=(1-train_size)
        )
    )

    if not independent_variables:
        independent_variables = list(df.columns)
        independent_variables.remove(dependent_variable)
    logging.info(
        'Independent Variables for Modeling: {independent_variables}'.format(
            independent_variables=independent_variables
        )
    )

    model_variables = independent_variables + list(dependent_variable)
    df = df[model_variables]
    train, test =\
        train_test_split(df, test_size=(1-train_size))

    train_y_observed = train[dependent_variable]
    test_y_observed  = test[dependent_variable]
    train_x = train[independent_variables]
    test_x  = test[independent_variables]

    return train_x, test_x, train_y_observed, test_y_observed


def split_dataframe_by_column(df, column):
    """
        Purpose:
            Split dataframe into multipel dataframes based on uniqueness
            of columns passed in. The dataframe is then split into smaller
            dataframes, one for each value of the variable.
        Args:
            df (Pandas DataFrame): DataFrame to split
            column (string): string of the column name to split on
        Return
            split_df (Dict of Pandas DataFrames): Dictionary with the
                split dataframes and the value that the column maps to
                e.g false/true/0/1
    """
    logging.info('Spliting Dataframes on Column')
    logging.info(
        'Columns to Split on: {column}'.format(column=column)
    )

    column_values = df[column].unique()

    split_df = {}
    for column_value in column_values:
        split_df[str(column_value)] = df[df[column] == column_value]

    return split_df
