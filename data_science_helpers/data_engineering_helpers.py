#!/usr/bin/env python3
"""
    Library for Dealing with redundant Data Engineering Tasks. This will include
    functions for tranforming dictionaries and PANDAS Dataframes
"""

# Python Library Imports
import sys
import os
import logging
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

###
# Alter DataFrame Functions
###

def remove_overly_null_columns(df, percentage_null=.25):
    """
        Purpose:
            Remove columns with the count of null values
            exceeds the passed in percentage. This defaults
            to 25%.
        Args:
            df (Pandas DataFrame): DataFrame to remove columns
                from
            percentage_null (float): Percentage of null values
                that will be the threshold for removing or
                keeping columns. Defaults to .25 (25%)
        Return
            df (Pandas DataFrame): DataFrame with columns removed
                based on thresholds
    """
    logging.info('Removing Overly Null Columns from DataFrame')
    logging.info(
        'Null Percentage Set to {percentage}'.format(
            percentage=percentage_null
        )
    )

    columns_to_drop = []
    for column in df.columns:
        total_null = df[column].isnull().sum()
        if (total_null / len(df.index)) > percentage_null:
            columns_to_drop.append(column)
            logging.info(
                'Dropping Columns {column} due to high null counts'.format(
                    column=column
                )
            )

    return df.drop(columns_to_drop, axis=1)


def remove_high_cardinality_numerical_columns(df, percentage_unique=1):
    """
        Purpose:
            Remove columns with the count of unique values
            matches the count of rows. These are usually
            unique identifiers (primary keys in a database)
            that are not useful for modeling and can result
            in poor model performance. percentage_unique
            defaults to 100%, but this can be passed in
        Args:
            df (Pandas DataFrame): DataFrame to remove columns
                from
            percentage_unique (float): Percentage of null values
                that will be the threshold for removing or
                keeping columns. Defaults to 1 (100%)
        Return
            df (Pandas DataFrame): DataFrame with columns removed
                based on thresholds
    """
    logging.info('Removing Unique Identifiers from DataFrame')
    logging.info(
        'Uniqueness Percentage Set to {percentage}'.format(
            percentage=percentage_unique
        )
    )

    columns_to_drop = []
    for column in df.columns:
        count_unique = df[column].nunique()
        if (count_unique / len(df.index)) >= percentage_unique:
            columns_to_drop.append(column)
            logging.info(
                'Dropping Columns {column} due to high uniqueness'.format(
                    column=column
                )
            )

    return df.drop(columns_to_drop, axis=1)


def remove_high_cardinality_categorical_columns(df, max_unique_values=20):
    """
        Purpose:
            Remove columns with the count of unique values
            for categorical columns are over a specified threshold.
            These values are difficult to transform into dummies,
            and would not work for logistic/linear regression.
        Args:
            df (Pandas DataFrame): DataFrame to remove columns
                from
            max_unique_values (int): Integer of unique values
                that is the threshold to remove column
        Return
            df (Pandas DataFrame): DataFrame with columns removed
                based on thresholds
    """
    logging.info(
        'Removing Categorical Columns with a large number of '
        'options from DataFrame'
    )
    logging.info(
        'Uniqueness Threshold Set to {max_unique_values}'.format(
            max_unique_values=max_unique_values
        )
    )

    columns_to_drop = []
    for column in get_categorical_columns(df):

        if df[column].nunique() > max_unique_values:
            columns_to_drop.append(column)
            logging.info(
                'Dropping Columns {column} due to too many '
                'possibilities for categorical dataset'.format(
                    column=column
                )
            )

    return df.drop(columns_to_drop, axis=1)


def remove_single_value_columns(df):
    """
        Purpose:
            Remove columns with a single value
        Args:
            df (Pandas DataFrame): DataFrame to remove columns
                from
        Return
            df (Pandas DataFrame): DataFrame with columns removed
    """
    logging.info('Removing Columns with One Value from DataFrame')

    columns_to_drop = []
    for column in df.columns:
        if df[column].nunique() == 1:
            columns_to_drop.append(column)
            logging.info(
                'Dropping Columns {column} with a single value'.format(
                    column=column
                )
            )

    return df.drop(columns_to_drop, axis=1)


def remove_quantile_equality_columns(df, low_quantile=.05, high_quantile=.95):
    """
        Purpose:
            Remove columns where the low quantile matches the
            high quantile (data is heavily influenced by outliers)
            and data is not well spread out
        Args:
            df (Pandas DataFrame): DataFrame to remove columns
                from
            low_quantile (float): Percentage quantile to compare
            high_quantile (float): Percentage quantile to compare
        Return
            df (Pandas DataFrame): DataFrame with columns removed
    """
    logging.info('Removing Columns with Equal Quantiles from DataFrame')
    logging.info(
        'Quantiles set to Low: {low} and High: {high}'.format(
            low=low_quantile,
            high=high_quantile
        )
    )

    quantiles_low = df.quantile(low_quantile)
    quantiles_high = df.quantile(high_quantile)

    columns_to_drop = []
    for column in get_numeric_columns(df):
        quantile_low = quantiles_low[column]
        quantile_high = quantiles_high[column]
        if quantile_low == quantile_high:
            columns_to_drop.append(column)

    return df.drop(columns_to_drop, axis=1)


def mask_outliers_numerical_columns(df, low_quantile=.05, high_quantile=.95):
    """
        Purpose:
            Update outliers to be equal to the low_quantile and
            high_quantile values specified.
        Args:
            df (Pandas DataFrame): DataFrame to update data
            low_quantile (float): Percentage quantile to set values
            high_quantile (float): Percentage quantile to set values
        Return
            df (Pandas DataFrame): DataFrame with columns updated
    """
    logging.info('Masking Outliers with Values in Quantiles')
    logging.info(
        'Quantiles set to Low: {low} and High: {high}'.format(
            low=low_quantile,
            high=high_quantile
        )
    )

    outliers_low = (df < low_quantiles)
    outliers_high = (df > high_quantiles)

    df = df.mask(outliers_low, low_quantiles, axis=1)
    df = df.mask(outliers_high, high_quantiles, axis=1)

    return df


def convert_categorical_columns_to_dummies(df, drop_first=True):
    """
        Purpose:
            Convert Categorical Values into Dummies. Will also
            remove the initial column being converted. If
            remove first is true, will remove one of the
            dummy variables to remove prevent multicollinearity
        Args:
            df (Pandas DataFrame): DataFrame to convert columns
            drop_first (bool): to remove or not remove a column
                from dummies generated
        Return
            df (Pandas DataFrame): DataFrame with columns converted
    """
    logging.info('Converting Categorical Columns into Dummies')

    for column in get_categorical_columns(df):
        dummies = pd.get_dummies(
            df[column], drop_first=drop_first,
            prefix=column, prefix_sep=':'
        )
        df.drop([column], axis=1, inplace=True)
        df = pd.concat([df, dummies], axis=1)

    return df


def ensure_categorical_columns_all_string(df):
    """
        Purpose:
            Ensure all values for Categorical Values are strings
            and converts any non-string value into strings
        Args:
            df (Pandas DataFrame): DataFrame to convert columns
        Return
            df (Pandas DataFrame): DataFrame with columns converted
    """
    logging.info('Ensuring Categorical Column Values are Strings')

    for column in get_categorical_columns(df):
        df[column] = df[column].astype(str)

    return df


def encode_categorical_columns_as_integer(df):
    """
        Purpose:
            Convert Categorical Values into single value
            using sklearn LabelEncoder
        Args:
            df (Pandas DataFrame): DataFrame to convert columns
        Return
            df (Pandas DataFrame): DataFrame with columns converted
    """
    logging.info('Converting Categorical Columns into Encoded Column')

    lable_encoder_object = LabelEncoder()
    for column in get_categorical_columns(df):
        column_encoder = lable_encoder_object.fit(df[column])
        df['LabelEncoded:{0}'.format(column)] =\
            column_encoder.transform(df[column])
        df.drop([column], axis=1, inplace=True)

    return df


def replace_null_values_numeric_columns(df, replace_operation='median'):
    """
        Purpose:
            Replace all null values in a dataframe with other
            values. Options include 0, mean, and median; the
            default operation converts numeric columns to
            median
        Args:
            df (Pandas DataFrame): DataFrame to remove columns
                from
            replace_operation (string/enum): operation to perform
                in replacing null values in the dataframe
        Return
            df (Pandas DataFrame): DataFrame with nulls replaced
    """
    logging.info(
        'Replacing Null Values of Numeric Columns with Operation: '
        '{replace_op}'.format(
            replace_op=replace_operation
        )
    )

    categorical_columns = get_categorical_columns(df)
    null_columns = get_columns_with_null_values(df)

    for column in (null_columns.keys() - categorical_columns):
        logging.info(
            'Filling Nulls in Column {column}'.format(column=column))
        if replace_operation == '0':
            df[column].fillna(0, inplace=True)
        elif replace_operation == 'median':
            df[column].fillna(df[column].median(), inplace=True)
        elif replace_operation == 'mean':
            df[column].fillna(df[column].mean(), inplace=True)

    return df


def replace_null_values_categorical_columns(df):
    """
        Purpose:
            Replace all null values in a dataframe with "Unknown"
        Args:
            df (Pandas DataFrame): DataFrame to remove columns
                from
            replace_operation (string/enum): operation to perform
                in replacing null values in the dataframe
        Return
            df (Pandas DataFrame): DataFrame with nulls replaced
    """
    logging.info(
        'Replacing Null Values of Categorical Columns with "Unknown"'
    )

    numeric_columns = get_numeric_columns(df)
    null_columns = get_columns_with_null_values(df)

    for column in (null_columns.keys() - numeric_columns):
        logging.info(
            'Filling Nulls in Column {column}'.format(column=column))
        df[column].fillna('Unknown', inplace=True)
        df[column].replace([np.nan], ['Unknown'], inplace=True)

    return df

###
# Describe DataFrame Functions
###

def get_categorical_columns(df):
    """
        Purpose:
            Returns the categorical columns in a
            DataFrame
        Args:
            df (Pandas DataFrame): DataFrame to describe
        Return
            categorical_columns (list): List of string
                names of categorical columns
    """
    logging.info('Getting Categorical from DataFrame')

    return list(set(df.columns) - set(get_numeric_columns(df)))


def get_numeric_columns(df):
    """
        Purpose:
            Returns the numeric columns in a
            DataFrame
        Args:
            df (Pandas DataFrame): DataFrame to describe
        Return
            numeric_columns (list): List of string
                names of numeric columns
    """
    logging.info('Getting Numeric from DataFrame')

    return list(set(df._get_numeric_data().columns))


def get_columns_with_null_values(df):
    """
        Purpose:
            Get Columns with Null Values
        Args:
            df (Pandas DataFrame): DataFrame to describe
        Return
            columns_with_nulls (dict): Dictionary where
                keys are columns with nulls and the value
                is the number of nulls in the column
    """
    logging.info('Getting Columns in DataFrame with Null Values')

    columns_with_nulls = {}
    for column in list(df.columns[df.isnull().any()]):
        columns_with_nulls[column] =\
            df[column].isnull().sum()

    return columns_with_nulls
