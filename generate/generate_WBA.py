"""
Description: 
    This script pre-processes WBA data

Update:
    Last updated in 06/2024
    Source: https://www.worldbenchmarkingalliance.org/nature-benchmark/ 

Output:
    A DataFrame containing the cleaned and consolidated WBA data with relevant indicators and scores.

NOTES:
    - Note that the structure of the data might change in future versions.
"""

import pandas as pd

from filepaths import PATH_TO_WBA
from utils import clean_variable_names

# Variables of interest (Shortlist)
VARIABLES_OF_INTEREST_WBA_INDICATOR_SHEET = (
    'NAT.B01', 'NAT.B02', 'NAT.B06', 'NAT.C02', 'NAT.C05', 'NAT.C07'
)
VARIABLES_OF_INTEREST_WBA_ELEMENT_SHEET = (
    'NAT.B03.ED', 'NAT.B05.EA', 'NAT.B05.EB', 'NAT.B05.EC', 'NAT.B05.EG', 'NAT.B05.EJ', 'NAT.C05.EA'
)

EXCLUDE_COLUMNS = ['company_name', 'isin', 'sedol']


def load_and_clean_data(sheet_name, variables_of_interest, value_column, id_column):
    '''
    A simple helper function to load and clean different sheets of WBA data.
    It is incorporated below in the generate_WBA function.
    '''
    df = pd.read_excel(PATH_TO_WBA, sheet_name=sheet_name)
    df = clean_variable_names(df)
    df = df.dropna(subset=['isin'])
    df = df[df[id_column].isin(variables_of_interest)]
    df_wide = df.pivot(index='isin', columns=id_column, values=value_column)
    df = df.drop_duplicates(subset=['isin'])
    df_wide = df_wide.merge(df[['isin', 'company_name', 'sedol']], on='isin', how='left')
    df_wide.columns = ['wba_' + str(col) if col not in EXCLUDE_COLUMNS else col for col in df_wide.columns]
    return df_wide


def generate_WBA():
    """
    Generates a consolidated DataFrame of WBA data by loading, cleaning, and merging data from multiple sheets.

    This function loads data from the WBA indicator scores sheet, element scores sheet, and scores and ranks sheet. 
    It cleans the data & selects the variables of interest.

    Returns:
        pd.DataFrame: A consolidated DataFrame containing the cleaned and merged WBA data.
    """
    print("Generating WBA data...")

    # Load and clean indicator data
    df_wba_indicator = load_and_clean_data(
        sheet_name='2023_Indicator Scores',
        variables_of_interest=VARIABLES_OF_INTEREST_WBA_INDICATOR_SHEET,
        value_column='indicator_score_(out_of_1)',
        id_column='indicator_code'
    )

    # Load and clean element data
    df_wba_element = load_and_clean_data(
        sheet_name='2023_Element Scores',
        variables_of_interest=VARIABLES_OF_INTEREST_WBA_ELEMENT_SHEET,
        value_column='element_score',
        id_column='element_code'
    )

    # Load and clean scores and ranks data
    df_wba_scores_ranks = pd.read_excel(
        PATH_TO_WBA,
        sheet_name='2023_Scores and Ranks',
        usecols=['MA2: Ecosystems and biodiversity', 'MA3: Social inclusion and community impact', 'ISIN', 'SEDOL',
                 'Company Name']
    )
    df_wba_scores_ranks = clean_variable_names(df_wba_scores_ranks)
    df_wba_scores_ranks = df_wba_scores_ranks.dropna(subset=['isin'])
    df_wba_scores_ranks.columns = [
        'wba_' + str(col) if col not in EXCLUDE_COLUMNS else col for col in df_wba_scores_ranks.columns
    ]

    # Merge all WBA datasets
    df_wba = df_wba_indicator.merge(df_wba_element, on='isin', how='outer')
    df_wba = df_wba.merge(df_wba_scores_ranks, on='isin', how='outer')

    # Consolidate columns sedol and company_name
    df_wba['sedol'] = df_wba[['sedol', 'sedol_x', 'sedol_y']].bfill(axis=1).iloc[:, 0]
    df_wba['company_name'] = df_wba[['company_name', 'company_name_x', 'company_name_y']].bfill(axis=1).iloc[:, 0]

    # Drop the original sedol and company_name columns
    df_wba = df_wba.drop(columns=['sedol_x', 'sedol_y', 'company_name_x', 'company_name_y'])

    return df_wba
