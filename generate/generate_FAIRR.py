"""
Description: 
    This script pre-processes the FAIRR data, which is an evaluation of the policies of protein producers.

Source: 
    https://www.fairr.org/tools/protein-producer-index 

NOTE:
    USER ACTION REQUIRED!
    - The FAIRR assessment is accessible to members only.
    - Please see link above, register and download the data.
    - A placeholder script based on 2022 data is provided below but needs to be updated with the 2023 data.
"""

from pathlib import Path

import pandas as pd

from filepaths import PATH_TO_FAIRR
from utils import clean_variable_names

variables_of_interest_FAIRR = [
    'company_', 'isin', 'dcf_target_-_soy',
    'dcf_target_-_cattle', 'engagement,_monitoring,_tracebility_-_soy',
    'engagement,_monitoring,_tracebility_-_cattle', 'feed_ingredients_&_conversion_ratios',
    'feed_innovation', 'ecosystem_impacts', 'def_score', 'wor_score'
]


def generate_FAIRR():
    """
    Generate a FAIRR DataFrame from downloaded FAIRR data.

    This function loads the FAIRR data, cleans the variable names, selects the variables of interest,
    renames certain columns for clarity, and adds a 'fairr_' prefix to all column names except for
    'company_name' and 'isin'. If the FAIRR data file is not found, it prints instructions on how to access
    and update the data file.

    Returns:
        pd.DataFrame: A DataFrame containing the cleaned and processed FAIRR data. If the data file is not found,
                      returns an empty DataFrame with columns specified in variables_of_interest_FAIRR plus
                      'company_name' and 'isin'.
    """

    if not Path(PATH_TO_FAIRR).exists():
        print("The FAIRR data file is not found.")
        print("Please register at https://www.fairr.org/tools/protein-producer-index to access the data.")
        print("Download the data and update the PATH_TO_FAIRR variable in filepaths.py.")
        return pd.DataFrame(columns=variables_of_interest_FAIRR + ['company_name', 'isin'])

    df_fairr = pd.read_excel(PATH_TO_FAIRR, sheet_name='SUMMARY')
    df_fairr = clean_variable_names(df_fairr)

    df_fairr = df_fairr[variables_of_interest_FAIRR]

    # Rename columns
    new_names = {
        'company_': 'company_name',
        'dcf_target_-_soy': 'df_target_soy',
        'dcf_target_-_cattle': 'df_target_cattle',
        'engagement,_monitoring,_tracebility_-_soy': 'engagement_soy',
        'engagement,_monitoring,_tracebility_-_cattle': 'engagement_cattle',
        'feed_ingredients_&_conversion_ratios': 'feed_ingredients_ratios',
        'def_score': 'deforestation_score',
        'wor_score': 'work_score'
    }

    df_fairr = df_fairr.rename(columns=new_names)

    # Add prefix
    exclude_columns = ['company_name', 'isin']
    df_fairr.columns = [
        'fairr_' + str(col) if col not in exclude_columns else col for col in df_fairr.columns
    ]

    return df_fairr
