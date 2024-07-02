"""
Description: 
    This script compiles and processes three aggregated SPOTT datasets (Palm Oil, Timber Pulp Paper, Natural Rubber)
    as well as the respective detailed assessment data

Update:
    Last updated in 05/2024
    Source: https://www.spott.org/ (log in required)

Output:
    This script generates a DataFrame as output.
"""

import numpy as np
import pandas as pd

from filepaths import PATH_TO_SPOTT_RUBBER, PATH_TO_SPOTT_PALM_OIL, PATH_TO_SPOTT_TIMBER_ETC, \
    PATH_TO_SPOTT_PALM_OIL_QUESTIONS, PATH_TO_SPOTT_TIMBER_ETC_QUESTIONS, PATH_TO_SPOTT_RUBBER_QUESTIONS
from utils import clean_variable_names

variables_of_interest_SPOTT = (
    'company', 'sedol', 'thomson_reuters_ticker',
    'sust_policy_score',
    'landbank_score',
    'cert_standards_score',
    'def_biodiv_score',
    'hcv_hcs_score',
    'soils_fire_score',
    'community_land_labour_score',
    'smallholders_suppliers_score',
    'gov_grievance_score',
    'parent_company',
    'rspo_member'
)

important_questions_palm_oil = (1, 2, 55, 57, 129, 130)
important_questions_timber = (1, 2, 52, 54, 130, 131)
important_questions_rubber = (1, 2, 55, 57, 130, 131)


def load_and_process_SPOTT_data(path_to_data):
    """
    Load and process SPOTT data from a given path.
    
    Args:
        path_to_data (str): path to SPOTT data
    
    Returns:
        pd.DataFrame
    """
    df = pd.read_csv(path_to_data)
    df = clean_variable_names(df)

    # Extracting year from date column
    df['year'] = df['date'].str[-2:]
    df['year'] = '20' + df['year']
    df['year'] = pd.to_numeric(df['year'])

    # Sorting and getting the latest entry for each company
    df = df.sort_values(by='year', ascending=False).groupby(
        'company').first().reset_index()
    return df


def generate_SPOTT(vars_of_interest=variables_of_interest_SPOTT,
                   path_to_specific_questions=(PATH_TO_SPOTT_PALM_OIL_QUESTIONS,
                                               PATH_TO_SPOTT_TIMBER_ETC_QUESTIONS,
                                               PATH_TO_SPOTT_RUBBER_QUESTIONS),
                   questions_palm_oil=important_questions_palm_oil,
                   questions_timber=important_questions_timber,
                   questions_rubber=important_questions_rubber
                   ):
    """
    Generate a consolidated SPOTT DataFrame from multiple data sources.

    Args:
        vars_of_interest (tuple): Variables of interest to include in the output DataFrame.
        path_to_specific_questions (tuple): Tuple of paths to specific SPOTT question data.
        questions_palm_oil (tuple): Tuple of important palm oil questions.
        questions_timber (tuple): Tuple of important timber questions.
        questions_rubber (tuple): Tuple of important rubber questions.

    Returns:
        pd.DataFrame: A consolidated DataFrame containing SPOTT data.
    """
    print("Generating SPOTT data...")

    questions_palm_oil = list(questions_palm_oil)
    questions_timber = list(questions_timber)
    questions_rubber = list(questions_rubber)

    # Load and process the different datasets
    df_po = load_and_process_SPOTT_data(PATH_TO_SPOTT_PALM_OIL)
    df_nr = load_and_process_SPOTT_data(PATH_TO_SPOTT_RUBBER)
    df_timb = load_and_process_SPOTT_data(PATH_TO_SPOTT_TIMBER_ETC)

    # Checking if all dataframes have the same columns
    if not (df_po.columns.equals(df_nr.columns) and df_nr.columns.equals(df_timb.columns)):
        print('Not all dataframes have the same columns')
        return pd.DataFrame()

    # Concatenating datafranes
    df_spott = pd.concat([df_po, df_nr, df_timb], ignore_index=True)
    del df_po, df_nr, df_timb

    # Drop irrelevant columns
    irrelevant_columns = ['date', 'year', 'subsidiaries', 'bloomberg_ticker',
                          'location', 'sr_year', 'sustainability_report']
    df_spott.drop(columns=irrelevant_columns, inplace=True)

    # Keep only variables of interest to reduce clutter
    df_spott = df_spott[list(vars_of_interest)]

    for path_to_specific_questions in path_to_specific_questions:
        df_specific_questions = pd.read_csv(path_to_specific_questions, skiprows=7,
                                            usecols=['Company', 'Number', 'Points'])
        df_specific_questions = clean_variable_names(df_specific_questions)

        # Pivot the data
        df_specific_questions_wide = df_specific_questions.pivot_table(
            index='company', columns='number', values='points', aggfunc='first')
        if path_to_specific_questions == PATH_TO_SPOTT_PALM_OIL_QUESTIONS:
            df_specific_questions_palm_oil = df_specific_questions_wide[questions_palm_oil]
            prefix_po = 'palm_oil_'
            df_specific_questions_palm_oil = df_specific_questions_palm_oil.add_prefix(prefix_po)
        elif path_to_specific_questions == PATH_TO_SPOTT_TIMBER_ETC_QUESTIONS:
            df_specific_questions_timber = df_specific_questions_wide[questions_timber]
            prefix_ti = 'timber_'
            df_specific_questions_timber = df_specific_questions_timber.add_prefix(prefix_ti)
        elif path_to_specific_questions == PATH_TO_SPOTT_RUBBER_QUESTIONS:
            df_specific_questions_rubber = df_specific_questions_wide[questions_rubber]
            prefix_ru = 'rubber_'
            df_specific_questions_rubber = df_specific_questions_rubber.add_prefix(prefix_ru)

    # Merge dataframes
    df_spott_specific_questions = pd.merge(df_specific_questions_palm_oil, df_specific_questions_timber,
                                           on='company', how='outer')
    df_spott_specific_questions = pd.merge(df_spott_specific_questions, df_specific_questions_rubber,
                                           on='company', how='outer')
    del df_specific_questions_palm_oil, df_specific_questions_timber, df_specific_questions_rubber

    # Clean the data by replacing '-' with NaNs
    df_spott_specific_questions.replace('-', np.nan, inplace=True)

    # Merge with main SPOTT dataframe
    df_spott = pd.merge(df_spott, df_spott_specific_questions,
                        on='company', how='outer')

    # Rename company column into company_name
    df_spott = df_spott.rename(columns={'company': 'company_name'})

    # Add "spott_" prefix to all column names except exclude_columns
    exclude_columns = ['company_name', 'sedol', 'thomson_reuters_ticker']
    df_spott.columns = ['spott_' + col if col not in exclude_columns else col for col in df_spott.columns]

    # Remove duplicates
    df_spott = df_spott.drop_duplicates(subset=['company_name'])

    return df_spott
