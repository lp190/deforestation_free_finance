"""
Description: 
    This script pre-processes CDP Forestry data.
    Note that the user needs to provide the CDP data file!

Update:
    Last updated in 09/2023
    Source: CDP Forestry data
    
Outline of code:
    - Function #1: generate_cdp()
        This function prepares the CDP(Forestry) dataset. It uses the original cdp file, then loads the specific
        manually selected questions in and categorizes the given answers in order to rank their respective risk.
        
NOTES:
    NB. one could consider improving upon the following:
    - scaling method for risk assessment
    - consider classifying responses to question F4.5a_C2 for a more comprehensive analysis
    - optimize data loading and processing for efficiency
"""

# import packages
import os

import numpy as np
import pandas as pd

# import paths
from filepaths import PATH_TO_OUTPUT_FOLDER, PATH_TO_CDP_DATA


#### FUNCTION ####

def generate_cdp():
    """
    This function prepares the CDP(Forestry) dataset. It uses the original cdp file, then loads the specific
    manually selected questions in and categorizes the given answers in order to rank their respective risk.

    For now it loads the four following questions and answers in:
        1.) F1.7_C2_Indicate whether you have assessed the deforestation or conversion footprint for your disclosed commodities over the past 5 years, or since a specified cutoff date, and provide details. - Have you monitored or estimated your deforestation/conversion footprint?
        2.) F4.1_Is there board-level oversight of forests-related issues within your organization?
        3.) F4.5_Does your organization have a policy that includes forests-related issues?
        4.) F4.5a_C2_Select the options to describe the scope and content of your policy. - Content


    Returns:
        pd.DataFrame: A DataFrame containing the CDP policy risk variables of interest.
    """

    file_path = PATH_TO_CDP_DATA

    if not file_path.exists():
        print(f"File not found: {file_path}. Please provide the CDP data file.")
        return

    ## Question by Question Assessment

    ## Q1 // F1.7_C2 ##

    # Q1 {'sheet_name': 'F1.7', 'column_name': 'F1.7_C2_Indicate whether you have assessed the deforestation or conversion footprint for your disclosed commodities over the past 5 years, or since a specified cutoff date, and provide details. - Have you monitored or estimated your deforestation/conversion footprint?'},

    # Import from Excel
    sheet_name_q1 = 'F1.7'
    columns_to_read_q1 = [
        'Account number',
        'F1.7_C2_Indicate whether you have assessed the deforestation or conversion footprint for your disclosed commodities over the past 5 years, or since a specified cutoff date, and provide details. - Have you monitored or estimated your deforestation/conversion footprint?',
    ]
    df_q1 = pd.read_excel(file_path, sheet_name=sheet_name_q1,usecols=columns_to_read_q1)

    # Define the conditions and corresponding values for the new column
    conditions_q1 = [
        (df_q1[
             'F1.7_C2_Indicate whether you have assessed the deforestation or conversion footprint for your disclosed commodities over the past 5 years, or since a specified cutoff date, and provide details. - Have you monitored or estimated your deforestation/conversion footprint?'] == 'Yes, we estimate deforestation/conversion footprint based on sourcing area'),
        (df_q1[
             'F1.7_C2_Indicate whether you have assessed the deforestation or conversion footprint for your disclosed commodities over the past 5 years, or since a specified cutoff date, and provide details. - Have you monitored or estimated your deforestation/conversion footprint?'] == 'No, and we do not plan to monitor or estimate our deforestation/conversion footprint in the next two years'),
        (df_q1[
             'F1.7_C2_Indicate whether you have assessed the deforestation or conversion footprint for your disclosed commodities over the past 5 years, or since a specified cutoff date, and provide details. - Have you monitored or estimated your deforestation/conversion footprint?'] == 'Question not applicable'),
        (df_q1[
             'F1.7_C2_Indicate whether you have assessed the deforestation or conversion footprint for your disclosed commodities over the past 5 years, or since a specified cutoff date, and provide details. - Have you monitored or estimated your deforestation/conversion footprint?'] == 'Yes, we monitor deforestation/conversion footprint in our supply chain'),
        (df_q1[
             'F1.7_C2_Indicate whether you have assessed the deforestation or conversion footprint for your disclosed commodities over the past 5 years, or since a specified cutoff date, and provide details. - Have you monitored or estimated your deforestation/conversion footprint?'] == 'No, but we plan to monitor or estimate our deforestation/conversion footprint in the next two years')
    ]

    # VERSION 1 - RISK BASED
    # Assign values based on conditions
    values_q1 = [0, 1, 0, 0, 0.5]

    # Create a new column based on conditions
    df_q1['conversion_policy_risk'] = np.select(
        conditions_q1, values_q1, default=None)

    # VERSION 2 - POLICY IN PLACE OR NOT
    values_q1 = [1, 0, 0, 1, 0.3]
    df_q1['conversion_policy_exist'] = np.select(
        conditions_q1, values_q1, default=None)

    # Several answers are possible. The next 4 lines deal with it (could be improved)
    # Add a new variable containing the mean of the two variables by group
    df_q1['conversion_policy_exist_mean'] = df_q1.groupby('Account number')['conversion_policy_exist'].transform('mean')
    df_q1['conversion_policy_risk_mean'] = df_q1.groupby('Account number')['conversion_policy_risk'].transform('mean')

    # Delete the account number duplicates
    df_q1 = df_q1.drop_duplicates(subset=['Account number'])

    # replace original variables with the mean (and delete mean variables afterwards)
    df_q1['conversion_policy_exist'] = df_q1['conversion_policy_exist_mean']
    df_q1['conversion_policy_risk'] = df_q1['conversion_policy_risk_mean']
    del df_q1['conversion_policy_exist_mean']
    del df_q1['conversion_policy_risk_mean']

    ## Q2 // F4.1 ##
    # Q2 {'sheet_name': 'F4.1', 'column_name': 'F4.1_Is there board-level oversight of forests-related issues within your organization?'}

    # Import from Excel
    sheet_name_q2 = 'F4.1'

    columns_to_read_q2 = [
        'Account number',
        'F4.1_Is there board-level oversight of forests-related issues within your organization?',
    ]

    df_q2 = pd.read_excel(file_path, sheet_name=sheet_name_q2,
                          usecols=columns_to_read_q2)

    # Define the conditions and corresponding values for the new column
    conditions_q2 = [
        (df_q2['F4.1_Is there board-level oversight of forests-related issues within your organization?'] == 'Yes'),
        (df_q2['F4.1_Is there board-level oversight of forests-related issues within your organization?'] == 'No'),
        (df_q2[
             'F4.1_Is there board-level oversight of forests-related issues within your organization?'] == 'Question not applicable'),
    ]

    # VERSION 1 - RISK BASED
    # Assign values based on conditions
    values_q2 = [0, 1, 0]

    # Create a new column based on conditions
    df_q2['board_level_policy_risk'] = np.select(
        conditions_q2, values_q2, default=None)

    # VERSION 2 - Policy in place or not
    values_q2 = [1, 0, 0]
    df_q2['board_level_policy_exist'] = np.select(
        conditions_q2, values_q2, default=None)

    ## Q3 // F4.5 ##
    # Q3 {'sheet_name': 'F4.5', 'column_name': 'F4.5_Does your organization have a policy that includes forests-related issues?'}

    # Import from Excel
    sheet_name_q3 = 'F4.5'

    columns_to_read_q3 = [
        'Account number',
        'F4.5_Does your organization have a policy that includes forests-related issues?',
    ]

    df_q3 = pd.read_excel(file_path, sheet_name=sheet_name_q3,
                          usecols=columns_to_read_q3)

    # Define the conditions and corresponding values for the new column
    conditions_q3 = [
        (df_q3['F4.5_Does your organization have a policy that includes forests-related issues?']
         == 'Yes, we have a documented forests policy that is publicly available'),
        (df_q3['F4.5_Does your organization have a policy that includes forests-related issues?']
         == 'No, but we plan to develop one within the next two years'),
        (df_q3['F4.5_Does your organization have a policy that includes forests-related issues?']
         == 'Question not applicable'),
        (df_q3['F4.5_Does your organization have a policy that includes forests-related issues?']
         == 'Yes, we have a documented forests policy, but it is not publicly available'),
        (df_q3['F4.5_Does your organization have a policy that includes forests-related issues?'] == 'No'),
    ]

    # VERSION 1 - RISK BASED
    # Assign values based on conditions
    values_q3 = [0, 0.5, 0, 0.25, 1]

    # Create a new column based on conditions
    df_q3['deforestation_policy_risk'] = np.select(
        conditions_q3, values_q3, default=None)

    # VERSION 2 - POlicy in place or not
    values_q3 = [1, 0.3, 0, 0.8, 0]
    df_q3['deforestation_policy_exist'] = np.select(
        conditions_q3, values_q3, default=None)

    ## Q4 // F4.5a ##

    # Q4 {'sheet_name': 'F4.5a', 'column_name': 'F4.5a_C2_Select the options to describe the scope and content of your policy. - Content'}
    #                                           & 'F4.5a_C3_Select the options to describe the scope and content of your policy. - Please explain'
    # Import from Excel
    sheet_name_q4 = 'F4.5a'

    columns_to_read_q4 = [
        'Account number',
        'F4.5a_C2_Select the options to describe the scope and content of your policy. - Content',
    ]

    df_q4 = pd.read_excel(file_path, sheet_name=sheet_name_q4,
                          usecols=columns_to_read_q4)

    # Convert the column F4.5a_C2 into a string
    df_q4['F4.5a_C2_Select the options to describe the scope and content of your policy. - Content'] = \
        df_q4['F4.5a_C2_Select the options to describe the scope and content of your policy. - Content'].astype(
            str)

    # Classify the statements which appear in the content column.
    # 1.) Create a set to store all the unique statements
    # 2.) Categorize each statement in either good, bad or neutral (regarding deforestation)
    # 3.) Create a new variable which takes the average of all the statements in each column

    # Create a set to store unique statements
    unique_statements = set()

    # Iterate through each cell, split it into statements, and add each part to the set
    for cell in df_q4['F4.5a_C2_Select the options to describe the scope and content of your policy. - Content']:
        statements = cell.split(';')
        unique_statements.update(statements)

    # Convert the set to a list
    unique_statements_list = list(unique_statements)

    # Create a DataFrame from the list of unique Statements
    unique_statements_df = pd.DataFrame(
        {'Unique Parts': unique_statements_list})

    # Save the DataFrame to a CSV file
    unique_statements_df.to_csv(os.path.join(
        PATH_TO_OUTPUT_FOLDER,
        'robustness/unique_statements.csv'))

    # Only keep specific policy-risk variables and then merge on Account number
    df_cdp = pd.read_excel(file_path, sheet_name='Summary Data')
    df_q1 = df_q1[['Account number','conversion_policy_risk', 'conversion_policy_exist']]
    df_q2 = df_q2[['Account number','board_level_policy_risk', 'board_level_policy_exist']]
    df_q3 = df_q3[['Account number', 'deforestation_policy_risk','deforestation_policy_exist']]

    # Create a list of DataFrames to merge
    dfs_to_merge = [df_cdp, df_q1, df_q2, df_q3]
    
    # Merge DataFrames based on 'Account number' in a loop
    cdp_policy_risk = dfs_to_merge[0]  # Initialize with the first DataFrame
    for i in range(1, len(dfs_to_merge)):
        cdp_policy_risk = pd.merge(
            cdp_policy_risk, dfs_to_merge[i], on='Account number', how='outer')

    # Make sure ISINs is loaded as str
    cdp_policy_risk['ISINs'] = cdp_policy_risk['ISINs'].astype(str)

    # delete everything but the first ISIN in the ISINs variable
    # It still gives us more companies with a ISIN than using the variable "Primary ISIN" from cdp
    # Function to extract the first ISIN
    
    def extract_first_isin(isin_string):
        """
        This function transforms a string variable which might consists of two (or more) parts separated by a comma
        retrieves the first numerical value without whitespaces. If no comma is present it only removes whitespaces

        Args:
            isin_string: string variable where the first part up to the comma should be retrieved

        Returns:
            isin_string: Returns either the first part up to the comma
                         or if there is no comma the original variable entry (both stripped, so without whitespaces)
        """
        # todo: add the second ISIN as well in order to potentially marge based on both ISINs
        # Check if a comma is present in the string
        if ',' in isin_string:
            # Split the string and return the first ISIN
            return isin_string.split(',')[0].strip()
        else:
            # If no comma is present, return the original string
            return isin_string.strip()

    # Apply the function to the 'ISINs' column and create a new 'First_ISIN' column
    cdp_policy_risk['isin'] = cdp_policy_risk['ISINs'].apply(
        extract_first_isin)
    del cdp_policy_risk['ISINs'] # Drop ISINs variable

    # standardize all three score variables
    columns_to_standardize = ['conversion_policy_risk','board_level_policy_risk', 'deforestation_policy_risk']

    for col in columns_to_standardize:
        standardized_col = f"{col}_standardized"

        # Z-score standardization
        mean = cdp_policy_risk[col].mean()
        std = cdp_policy_risk[col].std()
        cdp_policy_risk[standardized_col] = (cdp_policy_risk[col] - mean) / std

        # Min-Max scaling
        min_val = cdp_policy_risk[standardized_col].min()
        max_val = cdp_policy_risk[standardized_col].max()
        cdp_policy_risk[standardized_col] = (
                                                    cdp_policy_risk[standardized_col] - min_val) / (max_val - min_val)

    # Only keep relevant variables
    cdp_policy_risk = cdp_policy_risk.loc[:,
                      ('isin', 'conversion_policy_risk', 'board_level_policy_risk', 'deforestation_policy_risk',
                       'conversion_policy_risk_standardized', 'board_level_policy_risk_standardized',
                       'deforestation_policy_risk_standardized',
                       'conversion_policy_exist', 'board_level_policy_exist', 'deforestation_policy_exist')]

    return cdp_policy_risk
