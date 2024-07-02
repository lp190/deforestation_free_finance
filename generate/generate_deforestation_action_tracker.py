"""
Description:
    This script processes the deforestation action tracker dataset. 
    It processes the total score column along with specific commodity scores and saves them together with the company names into a .csv file.

Update:
    Last updated in Q1 2024
    Source: https://globalcanopy.org/what-we-do/corporate-performance/deforestation-action-tracker/


NOTES:
- NB. one could implement the use of the subsidiary column in the score generation.
"""

import pandas as pd

from utils import clean_variable_names


def generate_dat_scores(path_to_deforestation_action_tracker):
    """
    Extracts variables of interest from the deforestation action tracker dataset.

    Args:
        path_to_deforestation_action_tracker (str): path to original xlsx deforestation action tracker file
        
    Returns:
        (pd.dataframe): several columns; one with the company name, one with the overall DAT score as well as several
                        columns regarding FIs human rights policies
    """
    print("Generating deforestation action tracker data...")

    # Read the specified columns from the deforestation action tracker Excel file
    columns_to_use = [
        'FI name', ' 1.1 Score', 'Palm oil 4.3 Score', 'Timber, Pulp Paper 4.3 Score',
        'Soy 4.3 Score', 'Beef Leather 4.3 Score', 'Total Score /100'
    ]
    dat_scores = pd.read_excel(path_to_deforestation_action_tracker, sheet_name=2, usecols=columns_to_use)

    # Clean variable names
    dat_scores = clean_variable_names(dat_scores)

    # Rename columns for netter readability
    dat_scores.rename(columns={'fi_name': 'company_name', 'total_score_/100': 'score'}, inplace=True)

    # Add "dat_" prefix to all column names except the exclude_columns
    exclude_columns = ['company_name', 'sedol', 'thomson_reuters_ticker']
    dat_scores.columns = [
        'dat_' + str(col) if col not in exclude_columns else col
        for col in dat_scores.columns
    ]

    return dat_scores
