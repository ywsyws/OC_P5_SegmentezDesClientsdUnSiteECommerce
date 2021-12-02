from math import ceil
from datetime import date

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import statsmodels.api as sm


### Data Cleaning


def compare_dfs(df1: pd.DataFrame, df2: pd.DataFrame) -> list:
    """
    check different columns of the 2 dataframes
    """
    # columns only exist in df1
    df1_only = list(set(df1.columns) - set(df2.columns))
    # columns only exist in df2
    df2_only = list(set(df2.columns) - set(df1.columns))
    print(df1_only, "\nTotal: ", len(df1_only), "columns\n")
    print(df2_only, "\nTotal: ", len(df2_only), "columns\n")
    return df1_only, df2_only


def null_visual(df: pd.DataFrame, num_rows: int=1000):
    fig, (ax1, ax2) = plt.subplots(2, figsize=(24, 24))
    # r'$\underline{sin(x)}$'
    fig.suptitle("Visualisation of Null Values", y=1.03, fontsize=30)

    if len(df) > 10000:
        msno.matrix(df.sample(num_rows), ax=ax1, sparkline=False)
        msno.bar(df.sample(num_rows), ax=ax2)
        # msno.matrix(df.sample(1000), ax=ax1, sparkline=False)
        # msno.bar(df.sample(1000), ax=ax2)
    else:
        msno.matrix(df, ax=ax1, sparkline=False)
        msno.bar(df, ax=ax2)
    plt.show()
    return


def clean_columns(df: pd.DataFrame, filled_rate: float = 0.6) -> pd.DataFrame:
    """
    Data cleansing for null columns

    Parameters
    ----------
    df : dataframe
        The dataframe to be cleaned

    filled_rate : float, default 0.6
        The filled rate to be kept for all columns

    Returns
    -------
    dataframe
        A cleaned dataframe with columns that are filled more than the 'filled_rate'
    """

    print(f"Initial shape of the dataframe: {str(df.shape) : >17}")
    # keep columns that are filled more than the filled rate, default = 60%
    df = df.loc[:, (df.isnull().mean() < (1 - filled_rate))]
    print(f"Shape after removing null columns: {str(df.shape) : >14}")

    return df


def clean_rows_na(df: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    """
    Data cleansing for null rows between columns col1 and col2

    Parameters
    ----------
    df : dataframe
        The dataframe to be cleaned
    col1, col2: str
        The start and end columns respectively
        indicating the columns to be filtered

    Returns
    -------
    dataframe
        A cleaned dataframe without rows that have null values
        between columns col1 and col2
    """

    # create mask to filter df with rows that have ONLY null values
    # in columns indicated in col1 and col2
    print(f"Original shape: {df.shape} \n")
    slice1 = df.columns.get_loc(col1)
    slice2 = df.columns.get_loc(col2)
    mask = [any(df.iloc[row, slice1 : slice2 + 1].notna()) for row in range(len(df))]
    # filter df
    df = df.iloc[mask]
    df.reset_index(drop=True, inplace=True)
    print(
        f"Shape after removing null rows between '{df.columns[slice1]}'\n and '{df.columns[slice2]}':\n"
    )
    print(df.shape, "\n")

    return df


def clean_na_rows_any(df: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    """
    Data cleansing for null rows between columns col1 and col2

    Parameters
    ----------
    df : dataframe
        The dataframe to be cleaned
    col1, col2: str
        The start and end columns respectively
        indicating the columns to be filtered

    Returns
    -------
    dataframe
        A cleaned dataframe without rows that have null values
        between columns col1 and col2
    """

    # create mask to filter df with rows that have null values
    # in any of the columns indicated in col1 and col2
    slice1 = df.columns.get_loc(col1)
    slice2 = df.columns.get_loc(col2)
    mask = [all(df.iloc[row, slice1 : slice2 + 1].notna()) for row in range(len(df))]
    # filter df
    df = df.iloc[mask]
    df.reset_index(drop=True, inplace=True)
    print(
        f"Shape after removing null rows between '{df.columns[slice1]}'\n and '{df.columns[slice2]}'':\n"
    )
    print(df.shape, "\n")

    return df


def clean_rows_cat_values(df: pd.DataFrame, col: str, values: list) -> pd.DataFrame:
    """
    Delete rows that have values indicated in the values list of the indicated column

    Parameters
    ----------
    df : dataframe
        The dataframe to be cleaned
    col: str
        A categorical variable colume to be looked up for
    values: list of str
        A list of categorical variable values (string) to be looked up for
        Rows will be deleted if the values of the indicated column
        match with one of the indicated values

    Returns
    -------
    dataframe
        A cleaned dataframe without rows having indicated values
    """

    # create mask to filter df with rows that have
    # the indicated values in the indicated column
    index = df.columns.get_loc(col)
    mask = [df.iloc[row, index] not in values for row in range(len(df))]

    # print original dataframe shape
    print(f"Shape of the original dataframe: \n{df.shape}\n")

    # filter df
    df = df.iloc[mask]
    df.reset_index(drop=True, inplace=True)
    print(
        f"Shape after removing rows with values equal to\n{values}\nin column '{col}'':"
    )
    print(df.shape, "\n")

    return df


def clean_rows_num_values(
    df: pd.DataFrame, cols: str, value: float, compare: str
) -> pd.DataFrame:
    """
    Filter OUT rows that have values smaller than, equal to or bigger than
    the indicated value of the indicated column

    Parameters
    ----------
    df : dataframe
        The dataframe to be cleaned
    cols: str
        A list of numerical variable colume to be looked up for
    value: float
        A value to be compared with the values of the indicated column
        Rows will be deleted if the values of the indicated column
        match with the indicated comparison method of the indicated value
    compare: str
        A comparison method to be used to compare the values of the indicated column
        with the compared value provided


    Returns
    -------
    dataframe
        A cleaned dataframe without rows that match with
        the indicated comparison method of the indicated value
    """

    # print original dataframe shape
    print(f"Shape of the original dataframe: \n{df.shape}\n")

    # filter dataframe according to corresponding comparison method and value
    if compare == "equal":
        for col in cols:
            df = df[df[col] != value]
    elif compare == "bigger":
        for col in cols:
            df = df[df[col] <= value]
    elif compare == "smaller":
        for col in cols:
            df = df[df[col] >= value]
    elif compare == "bigger_equal":
        for col in cols:
            df = df[df[col] < value]
    elif compare == "smaller_equal":
        for col in cols:
            df = df[df[col] > value]
    else:
        print("Not a valid comparison method.")
        print(
            "Valid methods are 'equal', 'bigger', 'smaller', 'bigger_equal', 'smaller_equal'.\n"
        )
        return

    df.reset_index(drop=True, inplace=True)
    print(
        f"Shape after removing rows satisfying the comparison condition in columns '{cols}'':"
    )
    print(df.shape, "\n")

    return df


def drop_duplicates(df: pd.DataFrame, subset: list = None) -> pd.DataFrame:
    """
    Data cleansing for duplicated rows

    Parameters
    ----------
    df : dataframe
        The dataframe to be cleaned

    subset : list of strings, optional (default is None)
        column label or sequence of labels, optional
        Only consider certain columns for identifying duplicates, by
        default use all of the columns.

    Returns
    -------
    dataframe
        A cleaned dataframe without duplicated rows
    """

    # drop duplicates if there is any

    df_sub = df[subset] if subset else df
    if df_sub.duplicated().any():
        df.drop_duplicates(subset=subset, keep="last", inplace=True)
        print(f"Shape after dropping duplicated rows:\n{df.shape}\n")
    else:
        print("There isn't duplicated data.")

    return df


def convert_to_log(df: pd.DataFrame, cols: list, drop=False) -> pd.DataFrame:
    suffix = '_log'
    for col in cols:
        new_col = col + suffix
        if (df[col] == 0).any():
            df[new_col] = np.log(df[col] + 1)
        else:
            df[new_col] = np.log(df[col])
    if drop:
        df.drop(columns=cols, inplace=True)
    return df


### Data Analysis


def create_quanti_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create dataframe that has only quantitative data

    Parameters
    ----------
    df : dataframe
        The dataframe that contains both qualitative and quantitative columns

    Returns
    -------
    df: dataframe
        Dataframe with only quantitative data

    """

    # create a dictionary that contains datatype of each column
    dtypeDict = dict(df.dtypes)
    # create a list of column names that contains only quantitative data
    quanti_cols = [
        key
        for key, value in dtypeDict.items()
        if value == "float64" or value == "int64" or value == "uint8"
    ]
    df = df[quanti_cols]
    return df


def create_quali_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create dataframe that has only qualitative data

    Parameters
    ----------
    df : dataframe
        The dataframe that contains both qualitative and quantitative columns

    Returns
    -------
    df : dataframe
        Dataframe with only qualitative data

    """

    # create a dictionary that contains datatype of each column
    dtypeDict = dict(df.dtypes)
    # create a list of column names that contains only quantitative data
    quali_cols = [key for key, value in dtypeDict.items() if value == "object"]
    df = df[quali_cols]

    return df


def check_unique(df):
    """
    Print out number of unique values for each column of a dataframe

    Parameters
    ----------
    df : dataframe
        The dataframe to be checked on unique values
    """

    print("Number of unique values for each column")
    print("=======================================")
    # print number of unique values of each column
    for col in df.columns:
        print(f"{col}: {df[col].nunique()}")


def list_unique(df):
    """
    Print out unique values for each column of a dataframe

    Parameters
    ----------
    df : dataframe
        The dataframe to be checked on unique values
    """

    # print unique values of each column
    for col in df.columns:
        print(f"{col}:")
        print(f"{list(df[col].unique())}\n")


def replace_duplicate_values(df: pd.DataFrame, values: dict) -> pd.DataFrame:
    for key, value in values.items():
        df[key].replace(value, inplace=True)
    return df


def plot_boxplots(
    df: pd.DataFrame, drop_cols: list = None, sub_col=3, figsize: tuple = (18, 26)
):
    """
    Plot boxplot of a dataframe

    Parameters
    ----------
    df : dataframe
        The dataframe to be plotted
    drop_cols: list[str], optional (default is None)
        A list of string indicating a the column names to be dropped
    sub_col : int, optional (default is 3)
        Number of columns of subplots,
        i.e., number of subplots in each row
    figsize: tuple, optional (default is (18, 26))
        figsize represents the size of the figure

    Returns
    -------
    boxplots
        Boxplots of the input dataframe in the form of subplots

    """

    # drop unnecessary columns
    if drop_cols:
        df = df.drop(drop_cols, axis=1)
    # keep only quantitative features
    df = create_quanti_df(df)
    print(f"Number of quantitaive columns: {df.shape[1]}")
    # create figure and axes based on the number of columns of the dataframe
    fig, axes = plt.subplots(ceil(len(df.columns) / sub_col), sub_col, figsize=figsize)
    y = 0  # set counter

    # plot boxplot for each column of data
    for col in df.columns:
        i, j = divmod(y, sub_col)
        sns.boxplot(x=df[col], ax=axes[i, j]).set_title(col, fontsize=20)
        y += 1

    plt.tight_layout()
    plt.show()
    return


def plot_histplots(
    df: pd.DataFrame,
    var_type: str = "quant",
    drop_cols: list = None,
    figsize=(15, 20),
    sub_col=3,
    ticksize=15,
    div: int = 1,
    subplot=True,
) -> sns.histplot:
    """
    Plot histogram of a dataframe with only quantitative variables

    Parameters
    ----------
    df : dataframe
        The dataframe to be plotted
    var_type: string, {'quant', 'qual'} (default is 'quant')
        To indicate what kind of variables to be plotted
        - 'quant': plot quantitative variables
        - 'qual': plot qualitative varaibles
    drop_cols: list[str], optional (default is None)
        A list of string indicating a the column names to be dropped
    figsize: tuple, optional (default is (15, 20))
        figsize represents the size of the figure
    sub_col : int, optional (default is 3)
        Number of columns of subplots,
        i.e., number of subplots in each row
    ticksize : int, optional (default is 15)
        Fontsize of y ticks
    div : int, optional (default is 1)
        In case the size of the dataframe is too big,
        the length of the dataframe will be divided by div
        in order to show less data and save calculation time
    subplot : bool, optional (default is True)
        To indicate whether to plot subplots

    Returns
    -------
    histograms
        Histograms of the input dataframe in the form of subplots
    """

    assert var_type == "quant" or "qual", "var_type has to be either 'quant' or 'qual'."

    def print_error():
        print(f"Input var_type: {var_type} is invalid.")
        print("Valide var_type can only be 'quant' or 'qual'.")
        return

    def print_col():
        print(f"Number of {var_type}itaive columns: {df.shape[1]}")
        return

    def create_fig():
        # create figure and axes based on the number of columns of the dataframe
        _, axes = plt.subplots(
            ceil(len(df.columns) / sub_col), sub_col, figsize=figsize
        )
        y = 0  # set counter
        return axes, y

    if not subplot:
        # plt.figure(figsize=figsize)
        if var_type == "quant":
            sns.histplot(x=df)
        elif var_type == "qual":
            sns.histplot(y=df)
        else:
            print_error()

    else:
        # drop unnecessary columns
        if drop_cols:
            df = df.drop(drop_cols, axis=1)

        # create relative dataframe according to the var_type
        if var_type == "quant":
            # keep only quantitative features
            df = create_quanti_df(df)
            print_col()
            axes, y = create_fig()
            # plot histplot for each column of data
            for col in df.columns:
                i, j = divmod(y, sub_col)
                # sns.histplot(x=df[col], ax=axes[i, j]).set_title(col, fontsize=20)
                sns.histplot(x=df[col][: int(len(df) / div)], ax=axes[i, j]).set_title(
                    col, fontsize=20
                )
                y += 1
        elif var_type == "qual":
            # keep only qualitatve features
            df = create_quali_df(df)
            print_col()
            axes, y = create_fig()
            # plot histplot for each column of data
            for col in df.columns:
                i, j = divmod(y, sub_col)
                ax = axes[i, j]
                sns.histplot(y=df[col], ax=ax)
                ax.set_title(col, fontsize=20)
                ax.tick_params(axis="y", which="major", labelsize=ticksize)
                y += 1
        else:
            print_error()

    plt.tight_layout()
    plt.show()
    return


# def plot_histplots_quali(
#     df: pd.DataFrame, figsize=(15, 20), sub_col=3, ticksize=15, subplot=True
# ) -> sns.histplot:
#     """
#     Plot histogram of a dataframe with only qualitative variables

#     Parameters
#     ----------
#     df : dataframe or series
#         The dataframe to be plotted
#     figsize : tuple, optional
#         figure size
#         (default is (15, 20))
#     sub_col : int, optional
#         Number of columns of subplots,
#         i.e., number of subplots in each row
#         (default is 3)
#     ticksize : int, optional
#         Fontsize of y ticks
#         (default is 15)
#     subplot : bool, optional
#         To indicate whether to plot subplots
#         (default is True)

#     Returns
#     -------
#     histograms
#         Histograms of the input dataframe in the form of subplots
#     """

#     if not subplot:
#         plt.figure(figsize=figsize)
#         sns.histplot(y=df)

#     else:
#         # create figure and axes based on the number of columns of the dataframe
#         fig, axes = plt.subplots(
#             ceil(len(df.columns) / sub_col), sub_col, figsize=figsize
#         )
#         y = 0  # set counter

#         # plot histplot for each column of data
#         for col in df.columns:
#             i, j = divmod(y, sub_col)
#             ax = axes[i, j]
#             sns.histplot(y=df[col], ax=ax)
#             ax.set_title(col, fontsize=20)
#             ax.tick_params(axis="y", which="major", labelsize=ticksize)
#             y += 1

#     plt.tight_layout()
#     plt.show()
#     return


def plot_barplot(
    df: pd.DataFrame,
    feature: str,
    length: int = None,
    cutoff: float = None,
    figsize: tuple = (5, 10),
    ticksize: int = 15,
) -> pd.DataFrame:
    """
    Count and rank the frequency of each value of a categorical variable.
    Then plot it with barplot

    Parameters
    ----------
    df : dataframe
        The dataframe to be plotted
    feature : string
        The categorical variable in the dataframe to be plotted
    length : integer, optional (default is None)
        In case of too many categories within the feature chosen,
        the value of length decides how many top values will be displayed
    cutoff : float, optional (default is None)
        To indicate the cutoff line position in the y-axis
    figsize : tuple, optional (default is (5, 10))
        figure size
    ticksize : integer, optional (default is 15)
        font size of ytick labels

    Returns
    -------
    barplot
        Barplot of the chosen feature of the input dataframe
    dataframe
        Dataframe contains the ranked frequency of the indicated categorical variable.
    """

    # count each category within the feature
    count = df.groupby(f"{feature}")[f"{feature}"].count().sort_values(ascending=False)
    # convert it into a dataframe
    df_count = pd.DataFrame(columns=(["counts"]))
    df_count.counts = count

    # plot barplot
    plt.figure(figsize=figsize)
    x = df_count.counts[:length]
    y = df_count.index[:length]
    b = sns.barplot(x=x, y=y)
    b.set_yticklabels(df_count.index[:length], size=ticksize)
    # add a cut off line
    if cutoff:
        plt.axhline(y=cutoff, linestyle="--")
    plt.show()
    return df_count


def group_to_others(
    df: pd.DataFrame, to_be_grouped: dict, replace_value: str = "Other"
) -> pd.DataFrame:
    """
    Keep the indicated values of a categorical variable and
    group the rest into a value (default is "Other")

    Parameters
    ----------
    df : dataframe
        The dataframe to be grouped

    to_be_grouped : dict
        A dictionary contains:
        keys: categorical variables to be grouped
        values: a list of values to be kept
    replace_value: str
        All the values that are not in the list of values
        will be grouped to a value indicated in replaced_value
        (default is "Other")

    Returns
    -------
    dataframe
        A dataframe with categorical variable values grouped
    """

    for feature, values in to_be_grouped.items():
        df[feature] = [row if row in values else replace_value for row in df[feature]]
    return df


def heatmap(
    df: pd.DataFrame, figsize: tuple = (10, 10), scale: float = 1.4
) -> sns.heatmap:
    """
    Plot heatmap of a dataframe

    Parameters
    ----------
    df : dataframe
        The dataframe to be plotted
    figsize : tuple, optional
        figure size
        (default is (10, 10))

    Returns
    -------
    heatmap
        A heatmap that shows the quantitaive features correlations
        of the input dataframe
    """

    # keep only quantitative features
    df = create_quanti_df(df)
    print(f"Number of quantitaive columns: {df.shape[1]}")
    # calcaulate features correlations
    corr = df.corr() * 100
    # create mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # create figure
    plt.figure(figsize=figsize)
    # create heatmap
    sns.set(font_scale=scale)
    sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm", fmt=".0f")
    sns.reset_orig()
    plt.show()
    return


def create_quanti_cols(df: pd.DataFrame) -> list:
    """
    Create a list of column names that has only quantitative data

    Parameters
    ----------
    df : dataframe
        The dataframe that contains both qualitative and quantitative columns

    Returns
    -------
    quanti_cols: list
        List with column names that have quantitative data

    """

    # create a dictionary that contains datatype of each column
    dtypeDict = dict(df.dtypes)
    # create a list of column names that contains only quantitative data
    quanti_cols = []
    quali_cols = []
    for key, value in dtypeDict.items():
        if value == "float64" or value == "int64" or value == "uint8":
            quanti_cols.append(key)
        elif value == "object" or value == "bool":
            quali_cols.append(key)
        else:
            print(f"No such dtypes values yet. Please add {value} in the function")
    if len(quali_cols) == 1:
        return quanti_cols, quali_cols[0]
    else:
        return quanti_cols, quali_cols


def plot_category_boxplots(
    df: pd.DataFrame,
    figsize: tuple,
    length: int = None,
    method: str = None,
) -> sns.boxplot:
    """
    Plot boxplots against categories on the y-axis

    Parameters
    ----------
    df : dataframe
        The dataframe to be plotted
    figsize: tuple
        figsize represents the size of the figure
    length : integer, optional (default is None)
        In case of too many categories within the feature chosen,
        the value of length decides how many top values will be displayed
    method: str, optional (default is None)
        method indicates the way the categories to be ranked:
        by median, by mean or by quantile 75%

    Returns
    -------
    boxplot
        Boxplots showing different quantitative features against category,
        in the form of subplots
    """

    # check if the method input is valid
    if method not in ["median", "mean", "quantile75", None]:
        print("Not a valid method.")
        print("Valid methods are 'median', 'mean' and 'quantile75'.\n")
        return

    # create column list that contains only quantitative features
    quanti_cols_list, quali_col = create_quanti_cols(df)

    # create figure and axes for the subplots
    fig, axes = plt.subplots(len(quanti_cols_list), 1, figsize=figsize)

    # iterate over quantiative features
    # and plot boxplots against category
    for index, value in enumerate(quanti_cols_list):
        # Determine the order of boxes by median
        if method == "median":
            order = (
                df.groupby(by=[quali_col])[value]
                .median()
                .sort_values(ascending=False)
                .index
            )
        # Determine the order of boxes by mean
        elif method == "mean":
            order = (
                df.groupby(by=[quali_col])[value]
                .mean()
                .sort_values(ascending=False)
                .index
            )
        # Determine the order of boxes by quantile 75%
        elif method == "quantile75":
            order = (
                df.groupby(by=[quali_col])[value]
                .quantile(0.75)
                .sort_values(ascending=False)
                .index
            )
        else:
            order = None

        # plot boxplot
        sns.boxplot(
            y=df[quali_col],
            x=df[value],
            order=order[:length],
            showfliers=False,
            ax=axes[index],
        )

    plt.tight_layout()
    return


def group_quanti_values(
    df: pd.DataFrame,
    col: str,
    new_col: str,
    ranges: list,
    year: bool = False,
) -> pd.DataFrame:
    """
    Group the values of a quantitaive variable into different ranges


    Parameters
    ----------
    df : dataframe
        The dataframe to be manipulated.
    col: str
        The name of the quantitaive variable to be grouped.
    new_col: str
        The name of the new columne to store the grouped values.
    ranges: list
        A list of numbers to set the ranges to be grouped.
    year: bool, optional
        If year is True, the difference of the year of the data
        in each row and the year todate will be calculated.
        This difference will be served to put the entries
        into corresponding range.
        If year is False, the data of each row will be taken
        as it is to put the entries into corresponding range.
        (default is False)


    Returns
    -------
    dataframe
        A dataframe with an additional column that groups
        the data in the indicated column into the indicated ranges.
    """

    # calculate the year todate
    if year:
        year_todate = date.today().year
    for row in range(len(df)):
        # calculate the difference of the year todate and the value in the data
        if year:
            anchor = year_todate - df[col][row]
        else:
            anchor = df[col][row]
        # put the value of each row into its corresponding range
        if anchor < ranges[0]:
            df.loc[row, new_col] = f"<{ranges[0]}"
            pass
        i = 0
        while i < len(ranges) - 1:
            if anchor >= ranges[i] and anchor < ranges[i + 1]:
                df.loc[row, new_col] = f"{ranges[i]}-{ranges[i+1]}"
                break
            i += 1
        if anchor >= ranges[len(ranges) - 1]:
            df.loc[row, new_col] = f">{ranges[i]}"

    return df


def plot_catplot(
    df: pd.DataFrame,
    cat_col: str,
    quanti_cols_list: list,
    order: list = None,
    h: int = 5,
    w: int = 10,
) -> sns.catplot:
    """
    Plot caplots with the catagorical variable on the y-axis

    Parameters
    ----------
    df : dataframe
        The dataframe to be plotted
    cat_col : str
        The name of the catagorical variable to be plotted on the y-axis
    quanti_cols_list: list
        A list of numerical variables to be plotted on the x-axis
    order: list
        The order of the catagorical variable to be plotted
        (default is None)
    h, w: int
        The height and weight of each graph
        (default is 5 and 10)

    Returns
    -------
    catplot
        Catplots showing the categorical variable plotted against
        different quantitative variables
    """

    # iterate over quantiative variables
    # and plot catplot against the category variable
    for col in quanti_cols_list:
        sns.catplot(
            data=df,
            kind="bar",
            x=cat_col,
            y=col,
            order=order,
            ci=None,
            height=h,
            aspect=w / h,
        )
    plt.show()
    return

def plot_scree(df:pd.DataFrame, pca:PCA) -> plt.plot:
    # get explained variance ratio in percentage
    scree = pca.explained_variance_ratio_ * 100

    # plot barchart for each explained variance ratio
    plt.bar(np.arange(len(scree)) + 1, scree, label="PCA (individual)")

    # plot accumulated explained variance
    plt.plot(
        np.arange(len(scree)) + 1,
        scree.cumsum(),
        c="red",
        marker="o",
        label="PCA (accumulated)",
    )

    plt.title("PCA Scree Plot")
    plt.xlabel("Number of principal components")
    plt.ylabel("Variance Explained (%)")
    plt.legend()
    plt.show()
    return

def plot_pca_circle(df:pd.DataFrame, pca:PCA) -> plt.plot:
    # Plot a variable factor map for the first two dimensions.
    (fig, ax) = plt.subplots(figsize=(8, 8))
    for i in range(0, pca.components_.shape[1]):
        ax.arrow(0,
                0,  # Start the arrow at the origin
                pca.components_[0, i],  #0 for PC1
                pca.components_[1, i],  #1 for PC2
                head_width=0.1,
                head_length=0.1)

        plt.text(pca.components_[0, i] + 0.05,
                pca.components_[1, i] + 0.05,
                df.columns.values[i])


    an = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale
    plt.axis('equal')
    ax.set_title('Variable factor map')
    plt.show()

# define a function to calculate VIF
def calculate_vif(data: pd.DataFrame) -> pd.DataFrame:
    vif_df = pd.DataFrame(columns = ['Variable', 'Vif'])
    x_var_names = data.columns
    for i in range(0, x_var_names.shape[0]):
        y = data[x_var_names[i]]
        x = data[x_var_names.drop([x_var_names[i]])]
        # in case if there are infinitive values in x
        x = pd.DataFrame(x.replace([np.inf, -np.inf], np.nan))
        x = x.fillna(method='ffill')
        x = x.fillna(method='bfill')
        x = x.iloc[:, 0]
        r_squared = sm.OLS(y,x.astype(float), missing='drop').fit().rsquared
        vif = round(1/(1-r_squared),2)
        vif_df.loc[i] = [x_var_names[i], vif]
    return vif_df.sort_values(by = 'Vif', axis = 0, ascending=False, inplace=False)