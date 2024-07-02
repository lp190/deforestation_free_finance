"""
Description: 
    This script pre-processes SBTI data

Update:
    Last updated in 06/2024
    Source: https://sciencebasedtargets.org/companies-taking-action

Output:
    This script generates a DataFrame as output.

NOTES:
    - If SBTI data is used for non-listed companies, don't delete ISINs.
"""

import pandas as pd

from filepaths import PATH_TO_SBTI
from utils import clean_variable_names

variables_of_interest_SBTI = ('company_name', 'isin', 'lei',
                              'long_term_-_target_status', 'near_term_-_target_status')


def generate_SBTI(vars_of_interest=variables_of_interest_SBTI):
    """
    Generate SBTI DataFrame by reading an Excel file, cleaning the data, and performing various transformations.

    Args:
        vars_of_interest (tuple, optional): tuple of variables of interest to keep in the DataFrame.
                                                Defaults to vars_of_interest_SBTI.

    Returns:
        pandas.DataFrame: The generated SBTI DataFrame.

    """
    print("Generating SBTI data. Consider updating the file if needed.")

    df_sbti = pd.read_excel(PATH_TO_SBTI)
    df_sbti = clean_variable_names(df_sbti)

    # delete rows with missing ISIN codes
    df_sbti = df_sbti.dropna(subset=['isin'])
    df_sbti = df_sbti[df_sbti['isin'] != 'not available']
    df_sbti = df_sbti[df_sbti['isin'] != '.']

    # delete duplicate ISIN codes
    df_sbti = df_sbti.drop_duplicates(subset=['isin'])

    # keep only vars of interest
    df_sbti = df_sbti[list(vars_of_interest)]

    # rename columns
    df_sbti = df_sbti.rename(columns={'long_term_-_target_status': 'long_term_target',
                                      'near_term_-_target_status': 'near_term_target'})

    # get rid of rows with near_term_target = 'Removed'
    df_sbti = df_sbti[df_sbti['near_term_target'] != 'Removed']

    # add "sbti" prefix to all column names except exclude_columns
    exclude_columns = ['company_name', 'isin', 'lei']
    df_sbti.columns = [
        'sbti_' + str(col) if col not in exclude_columns else col for col in df_sbti.columns]

    return df_sbti
