"""
Description: 
    This script pre-processes Food Emissions data. The Food Emissions dataset assesses the 50 largest companies in the food sector on their emissions.
    We might be able to extract a few positive/negative flags, such as:
    - Metric 5d: Time-bound commitment to achieve a deforestation and conversion free supply chain by 2025 across the business
    - Metric 1b: Scope 3 from agriculture	
    - Metric 1c: Scope 3 from land use change

Update:
    Last updated in 10/2023
    Source: https://www.ceres.org/climate/ambition2030/food-emissions-50

Output:
    This script generates a DataFrame as output.
"""

import pandas as pd

from filepaths import PATH_TO_FOOD_EMISSIONS_50
from utils import clean_variable_names

variables_of_interest_FE_50 = (
    'company',
    'gics_sub-industry',
    'metric_1b:_scope_3_from_agriculture',
    'metric_1c:_scope_3_from_land_use_change',
    'metric_5d:_time-bound_commitment_to_achieve_a_deforestation_and_conversion_free_supply_chain_by_2025_across_the_business'
)


def generate_fe_50(vars_of_interest=variables_of_interest_FE_50):
    """
    Generates a DataFrame by pre-processing Food Emissions data.

    This function reads the Food Emissions 50 dataset, cleans the column names, selects the specified variables of
    interest, renames the 'company' column to 'company_name', and adds a 'fe50_' prefix to all column names except
    for the excluded columns.

    Args:
        vars_of_interest (tuple): A tuple of column names to be retained from the Food Emissions 50 dataset.
    
    Returns:
        pd.DataFrame: A DataFrame containing the pre-processed Food Emissions data with relevant columns.
    """
    print("Generating Food Emissions 50 data...")

    df_fe50 = pd.read_excel(PATH_TO_FOOD_EMISSIONS_50, sheet_name='data')
    df_fe50 = clean_variable_names(df_fe50)

    df_fe50 = df_fe50[list(vars_of_interest)]

    # add "fe50_" prefix to all column names except exclude_columns
    df_fe50 = df_fe50.rename(columns={'company': 'company_name'})
    exclude_columns = ['company_name', 'gics_subindustry']
    df_fe50.columns = [
        'fe50_' + str(col) if col not in exclude_columns else col for col in df_fe50.columns]

    return df_fe50
