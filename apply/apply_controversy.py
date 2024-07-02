"""
Filename: apply_controversy.py

Description:
    This script integrates "controversies", i.e., whether a company is involved in any controversies, into the portfolio data.
    As a default a flag is added whether the company is part of the F500 data.
    The script also allows to integrate controversy scores from other ESG data providers. 

NOTE:
    - If string matching is used, the accuracy could be improved.
    - The manual matches for the F500 data are currently only covering MSCI ACWI companies.
"""

import os

import pandas as pd

from filepaths import PATH_TO_OUTPUT_FOLDER
from prep.prep_forest500 import prep_forest500
from user_input import FUZZY_MATCH_CUTOFF_F500

# Specify the columns you want to load for forest500 analysis
columns_to_load_forest500_comps = ('Company', 'HQ', 'Total Score /100')
columns_to_load_forest500_fis = ('FI name', 'FI Headquarters', 'Total Score / 100')

# Manual overrule for incorrect matches (to rule out false positives)
# Note: this manual false positive list is generated for the MSCI_ACWI portfolio with a string matching cutoff of 75

manual_false_positive_list = ('bank of changsha', 'smc', 'shizuoka financial', 'misumi', 'rogersmmunications b',
                              'banco bradesco', 'china baoan', 'bank of chengdu', 'obic',
                              'sainsbury j', 'american financial', 'china citic bank')


def apply_controversy_filters(df_portfolio, esg_controversies=False,
                              forest500_columns=columns_to_load_forest500_comps,
                              forest500_fis_columns=columns_to_load_forest500_fis):
    """
    This function applies the controversy filter and adds the controversy scores as new column to the input dataframe.

    Args:
        df_portfolio: main dataframe containing the portfolio data
        esg_controversies: True/False statement whether controversy data is available
        forest500_columns:    List of columns which are retrieved from the raw forest500 dataset in order
                                            to do the matching and collect the respective company's score
        forest500_fis_columns:      List of columns which are retrieved from the raw forest500 dataset in order
                                            to do the matching and collect the respective FI's score


    Returns:
        df_portfolio (pd.DataFrame): The input dataframe with the controversy scores and flags added
                        "flag_forest500" - binary variable which equals 1 if company is on forest 500 list
                        (more subject to change depending on the data provider used for controversy scores)
    """

    # Store initial columns
    initial_columns = df_portfolio.columns

    # LOAD CONTROVERSY (incl F500) DATA IF ALREADY EXISTS, OTHERWISE PREPARE IT
    if os.path.exists(os.path.join(PATH_TO_OUTPUT_FOLDER, 'internal_data/forest500_scores.csv')):
        print('loading: controversy.csv')
        forest500_scores = pd.read_csv(os.path.join(PATH_TO_OUTPUT_FOLDER, 'internal_data/forest500_scores.csv'))
        print('DONE')
    else:
        print('generating: controversy.csv')
        print('preparing: forest500 controversy scores')
        forest500_scores = prep_forest500(df_portfolio, columns_comps=forest500_columns,
                                          columns_fis=forest500_fis_columns,
                                          fuzzy_match_cutoff=FUZZY_MATCH_CUTOFF_F500,
                                          false_positive_list=manual_false_positive_list,
                                          manual_matches=True)

    if esg_controversies:
        ### THE USER CAN INCORPORATE CONTROVERSY SCORES FROM ESG DATA PROVIDERS HERE ###
        print("ONLY A PLACEHOLDER - NO ESG CONTROVERSY DATA AVAILABLE YET!")

        # df_esg_controversies = prep_esg_controversies(df_esg_controversies)
        # df_portfolio = pd.merge(df_portfolio, df_esg_controversies, left_on=['permid'], right_on=['permid'], how='left')

    # Ensure that permid is in the right format
    forest500_scores['permid'] = forest500_scores['permid'].astype(str)

    ## Attach data to df_portfolio
    forest500_scores = forest500_scores[['permid', 'forest500_total_score', 'flag_forest500']]
    df_portfolio = pd.merge(df_portfolio, forest500_scores, left_on=['permid'], right_on=['permid'], how='left')

    # Save newly added columns for faster access
    new_columns = df_portfolio.columns.difference(initial_columns)
    controversy_data = df_portfolio[new_columns.tolist() + ['permid']]
    controversy_data.to_csv(os.path.join(PATH_TO_OUTPUT_FOLDER, 'internal_data/forest500_scores.csv'))

    # Print how many variables have been added

    print(f"New columns added: {', '.join(new_columns)}")

    for column in new_columns:
        non_zero_count = (df_portfolio[column] != 0).sum()
        print(f"Number of non-zero values in '{column}': {non_zero_count}")

    return df_portfolio
