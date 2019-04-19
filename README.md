# Christopher H. Todd's PROJECT_STRING_NAME

The PROJECT_GIT_NAME project is responsible for ...

The library ...

## Table of Contents

- [Dependencies](#dependencies)
- [Libraries](#libraries)
- [Example Scripts](#example-scripts)
- [Notes](#notes)
- [TODO](#todo)

## Dependencies

### Python Packages

- great-expectations>=0.4.5
- pandas>=0.24.2
- tensorflow>=1.13.1

## Libraries

### [data_engineering_helpers.py](https://github.com/ChristopherHaydenTodd/ctodd-python-lib-data-science/blob/master/data_science_helpers/data_engineering_helpers.py)

Library for Dealing with redundant Data Engineering Tasks. This will include functions for tranforming dictionaries and PANDAS Dataframes

Functions:

```
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
```

```
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
```

```
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
```

```
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
```

```
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
```

```
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
```

```
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
```

```
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
```

```
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
```

```
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
```

```
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
```

```
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
```


```
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
```


```
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
```

### [data_exploration_helpers.py](https://github.com/ChristopherHaydenTodd/ctodd-python-lib-data-science/blob/master/data_science_helpers/data_exploration_helpers.py)

Library for aiding the understanding and investigation into the data provided for modeling. These helpers will help explain, graph, and explore the data

Functions:

```
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
```


```
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
```


```
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
```


```
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
```


```
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
```

### [model_persistence_helpers.py](https://github.com/ChristopherHaydenTodd/ctodd-python-lib-data-science/blob/master/data_science_helpers/model_persistence_helpers.py)

Library for helping store/load/persist data science models using Python libraries

Functions:

```
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
```


```
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
```

### [model_training_helpers.py](https://github.com/ChristopherHaydenTodd/ctodd-python-lib-data-science/blob/master/data_science_helpers/model_training_helpers.py)

Library for helping train data science models using Python libraries

Functions:

```
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
```

```
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
```

## Example Scripts

Example executable Python scripts/modules for testing and interacting with the library. These show example use-cases for the libraries and can be used as templates for developing with the libraries or to use as one-off development efforts.

### N/A

## Notes

 - Relies on f-string notation, which is limited to Python3.6.  A refactor to remove these could allow for development with Python3.0.x through 3.5.x

## TODO

 - Unittest framework in place, but lacking tests
