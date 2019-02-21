#!/usr/bin/env python3
"""
    Library for aiding the understanding and investigation into the data provided
    for modeling. These helpers will help explain, graph, and explore the data
"""

# Python Library Imports
import sys
import os
import logging
import pandas as pd

from data_science_helpers.data_engineering_helpers import *

###
# Describe Data Functions
###

def get_numerical_column_statistics(df):
    """
        Purpose:
            Describe the numerical columns in a dataframe.
            This will include, total_count, count_null, count_0,
            mean, median, mode, sum, 5% quantile, and 95% quantile.
        Args:
            df (Pandas DataFrame): DataFrame to describe
        Return
            num_statistics (dictionary): Dictionary with key being
            the column and the data being statistics for the
            column
    """
    logging.info('Calculating Numerical Column Statistics')

    quantiles_5 = df.quantile(0.05)
    quantiles_25 = df.quantile(0.25)
    quantiles_50 = df.quantile(0.50)
    quantiles_75 = df.quantile(0.75)
    quantiles_95 = df.quantile(0.95)

    num_statistics = {}
    for column in get_numeric_columns(df):

        num_statistics[column] = {
            'quantile_5': quantiles_5[column],
            'quantile_25': quantiles_25[column],
            'quantile_50': quantiles_50[column],
            'quantile_75': quantiles_75[column],
            'quantile_95': quantiles_95[column],
            'mean': df[column].mean(),
            'median': df[column].median(),
            'max': df[column].max(),
            'min': df[column].min(),
            'sum': df[column].sum(),
            'skew': df[column].skew(),
            'std': df[column].std(),
            'var': df[column].var(),
        }

    return num_statistics

###
# Describe Column Correlation Functions
###

def get_column_correlation(df):
    """
        Purpose:
            Determine the true correlation between
            all column pairs in a passed in DataFrame.
            This is the pure correlation; this is useful
            if you are looking for the detailed correlation
            and the direction of the correlation
        Args:
            df (Pandas DataFrame): DataFrame to determine correlation
        Return
            unique_value_correlation (Pandas DataFrame): DataFrame
            of correlations for each column set in the DataFrame
    """
    logging.info('Getting Column Correlation')

    base_correlation = df.corr().unstack()
    unique_value_correlation =\
        base_correlation[get_unique_column_paris(df)]

    return unique_value_correlation


def get_column_absolute_correlation(df):
    """
        Purpose:
            Determine the absolute correlation between
            all column pairs in a passed in DataFrame.
            Absolute converts all correlations to a
            positive value; this is useful if you are
            only looking for the existance of a coorelation
            and not the direction.
        Args:
            df (Pandas DataFrame): DataFrame to determine correlation
        Return
            unique_value_abs_correlation (Pandas DataFrame): DataFrame
            of correlations for each column set in the DataFrame
    """
    logging.info('Getting Column Correlation (Absolute Value)')

    abs_correlation = df.corr().abs().unstack()
    unique_value_abs_correlation =\
        abs_correlation[get_unique_column_paris(df)]

    return unique_value_abs_correlation


def get_column_pairs_significant_correlation(df, pos_corr=.20, neg_corr=.20):
    """
        Purpose:
            Determine Columns with highly positive or highly
            negative correlation. Defaults for positive and
            negative correlations are 20% and can be passed
            in as parameters
        Args:
            df (Pandas DataFrame): DataFrame to determine correlation
            pos_corr (float): Float percentage to consider a positive
            correlation as significant. Default 20%
            neg_corr (float): Float percentage to consider a negative
            correlation as significant. Default 20%
        Return
            high_positive_correlation_pairs (List of Sets): List of column
            pairs with a high positive correlation
            high_negative_correlation_pairs (List of Sets): List of column
            pairs with a high negative correlation
    """
    logging.info('Getting Signification Correlation Column Pairs')
    logging.info('Positive Correlation Threshold: {0}'.format(pos_corr))
    logging.info('Negative Correlation Threshold: {0}'.format(neg_corr))

    correlation = get_column_correlation(df)
    unique_pairs = get_unique_column_paris(df)

    positive_correlation_pairs = []
    negative_correlation_pairs = []
    for pair in unique_pairs:
        if correlation[pair] >= pos_corr:
            positive_correlation_pairs.append(pair)
        if correlation[pair] <= neg_corr:
            negative_correlation_pairs.append(pair)

    return positive_correlation_pairs, negative_correlation_pairs

###
# Describe DataFrame Shape Functions
###

def get_unique_column_paris(df):
    """
        Purpose:
            Get unique pairs of columns from a DataFrame. This
            assumes there is no direction (A, B) and returns
            a Set of column pairs that can be used for identifying
            correlation, mapping columns, and other functions
        Args:
            df (Pandas DataFrame): DataFrame to determine column pairs
        Return
            unique_pairs (Set): Set of unique column pairs
    """
    logging.info('Getting Unique Column Pairs')

    unique_pairs = set()
    columns = df.columns

    for i in range(0, len(columns)):
        for j in range(i+1, len(columns)):
            unique_pairs.add((columns[i], columns[j]))

    return unique_pairs
