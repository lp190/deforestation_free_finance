"""
Filename: utils.py

Description:
    This script contains a collection of functions utilized across various parts of the project.
    The functions cover a wide range of utilities, including fuzzy string matching, data cleaning,
    standardizing company names, merging dataframes, and enriching data with geographical coordinates.

Overview:

  ##### HELPER FUNCTIONS FOR MERGING DATAFRAMES #####
  - `find_best_match`: Finds the best match for a given name within a list of choices using fuzzy string matching.
  - `find_best_match_open_source`: Fuzzy matching for open-source datasets.
  - `find_fuzzy`: Finds fuzzy matches between two lists using the rapidfuzz library.
  - `map_on_identifiers`: Maps IDs across dataframes based on specified identifiers.
  - `merge_and_count_matches`: Merges columns and counts the number of matches in each column.

  ##### HELPER FUNCTIONS FOR STANDARDISING COMPANY NAMES #####
  - `clean_company_name`: Cleans and standardizes company names by removing common patterns.
  - `clean_company_name_new`: Enhanced function to clean and standardize company names with additional patterns.
  - `standardize_company_names`: General function for standardizing company names.
  - `clean_variable_names`: Standardizes variable names in a DataFrame.

  ##### HELPER FUNCTIONS FOR MERGING ASSET LEVEL DATA #####
  - `map_on_identifiers`: Maps IDs across dataframes based on specified identifiers.
  - `merge_and_count_matches`: Merges columns and counts the number of matches in each column.

  ##### HELPER FUNCTIONS TO TURN DT1 DATA INTO A SCORING SYSTEM #####
  - `dt1_scoring_log_transform_and_normalize`: Log-transforms and normalizes a series.
  - `dt1_scoring_calculate_weighted_score`: Calculates the weighted score for a DataFrame based on a given weight dictionary.
  - `clean_df_portfolio`: Cleans and prepares the portfolio DataFrame for further processing.

  ##### HELPER FUNCTIONS THAT ARE NOT USED IN THE DEFAULT VERSION OF THIS REPOSITORY #####
  - `get_nace_details`: Extracts NACE macro sector and sector description at the 2-digit level.
  - `city2lonlat`: Enhances city data with longitude and latitude coordinates.
  - `fill_missing_location_values_cleaned_df`: Fills missing geographical coordinates using Google Maps API.
"""

# import packages
import os
import re

import googlemaps
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz, process
from rapidfuzz.distance import LCSseq

from filepaths import PATH_TO_INPUT_FOLDER


##### HELPER FUNCTIONS FOR MERGING DATAFRAMES #####
'''
Different functions to leverage fuzzy string matching to merge dataframes.
This could be streamlined in the future.
Rapidfuzz is the fastest library for fuzzy string matching.
'''


# FUZZY STRING MATCHING #1
def find_best_match(name, choices, score_cutoff):
    """
    Finds the best match for a given name within a list of choices using fuzzy string matching.

    This function utilizes the `process.extractOne` method from the `fuzzywuzzy` library
    to find the best match for a given name within a list of choices based on token sort ratio.

    Args:
        name (str): The input name to be matched against the choices.
        choices (list): A list of strings representing the available choices for matching.
        score_cutoff: Cutoff, needs to be a good enough match

    Returns:
        str or None: The best-matching choice from the list, or None if no match is found.

    Note:
        The `fuzzywuzzy` library must be installed to use this function.
    """
    return process.extractOne(
        name,
        choices,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=score_cutoff)

# Another helper function for fuzzy string matching

def find_best_match_open_source(row, choices, scorer=fuzz.token_sort_ratio, threshold=80):
    """"""
    best_match, score = process.extractOne(
        row['name_clean'], choices, scorer=scorer)
    if score >= threshold:
        return best_match
    return None


# FUZZY STRING MAPPING VIA RAPIDFUZZ
def find_fuzzy(list_master, list_asset_data, score_cutoff: int, scorer=fuzz.ratio):
    """
    Finds fuzzy matches between two lists using rapidfuzz library.
    Read this article for more context: https://medium.com/@bewin4u/fuzzy-matching-for-million-row-address-dataset-with-rapidfuzz-and-splink-b704eaf1fda9

    Args:
        list_master (list): Our master data (i.e, our universe)
        list_asset_data (list): The list of strings to compare against the master list (example: df_climate_trace_unmatched["standardized_parent_name"].tolist()
        score_cutoff (int): The minimum score required for a match to be considered.
        scorer (function): The scoring function to use. Defaults to fuzz.ratio.

    Returns:
        list: A list of dictionaries containing the matched pairs and their scores.
    """
    from rapidfuzz import process
    # Generate a score matrix using rapidfuzz
    score_matrix = process.cdist(
        list_master,
        list_asset_data,
        processor=lambda x: str(x).lower(),  # Ensure strings are lowered
        scorer=scorer,
        dtype=np.uint8,  # Output the score as uint8, which is faster
        workers=-1,  # Use multithreading. -1 means use all cores
        score_cutoff=score_cutoff,
    )

    results = []
    # Find non-zero elements in the score matrix (indicating matches above the cutoff)
    master_indices, asset_data_indices = np.nonzero(score_matrix)
    for master_index, asset_data_index in zip(master_indices, asset_data_indices):
        results.append({
            "df_master_index": master_index,
            "asset_data_index": asset_data_index,
            "df_master_name": list_master[master_index],
            "asset_data_name": list_asset_data[asset_data_index],
            "score_of_match": score_matrix[master_index, asset_data_index],
        })
    return results


###### HELPER FUNCTIONS FOR STANDARDISING COMPANY NAMES #####
'''
We have different functions here to standardize company names.
Sometimes they have been tailored to specific datasets.
Note that this could be streamlined in the future.
'''

def clean_company_name(text):
    """"""
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = text.replace('corporation', '')
    text = text.replace('holdings', '')
    text = text.replace('holding', '')
    text = text.replace('co inc', '')
    text = text.replace('incorporated', '')
    text = text.replace('inc.', '')
    text = text.replace('co ltd', '')
    text = text.replace('ltd', '')
    text = text.replace('co.', '')
    text = text.replace('limited', '')
    text = text.replace('corp', '')
    text = text.replace('s.a.', '')
    text = text.replace('plc', '')
    text = text.replace('a/s', '')
    text = text.replace('args', '')
    text = text.replace(' sa', '')
    text = text.replace(' group', '')
    text = text.replace(' ab', '')
    text = text.replace(' amba', '')
    text = text.replace(' ag', '')
    text = text.replace(' co', '')
    text = text.replace(' s a', '')
    text = text.replace(' b a', '')
    text = text.replace(' inc', '')
    text = text.replace(' international', '')

    text = re.sub('[\W_]+', ' ', text, flags=re.UNICODE)
    text = text.strip()

    return text


def clean_company_name_new(text):
    """"""
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = text.replace('polska', '')
    text = text.replace('corporación bancaria', '')
    text = text.replace(' of commerce', '')
    text = text.replace(' capital', '')
    text = text.replace('ag & co. kgaa', '')
    text = text.replace('the ', '')
    text = text.replace('italia', '')
    text = text.replace('nederland', '')
    text = text.replace('corporation', '')
    text = text.replace('holdings', '')
    text = text.replace('holding', '')
    text = text.replace(' and company limited', '')
    text = text.replace(' company limited', '')
    text = text.replace(' company', '')
    text = text.replace(' co inc', '')
    text = text.replace(' pjsc', '')
    text = text.replace(' p.j.s.c', '')
    text = text.replace('incorporated', '')
    text = text.replace(' namen', '')
    text = text.replace(', inc.', '')
    text = text.replace(', inc', '')
    text = text.replace('inc.', '')
    text = text.replace('ltd', '')
    text = text.replace('co.', '')
    text = text.replace(' kag', '')
    text = text.replace('limited', '')
    text = text.replace(' corp', '')
    text = text.replace(' s.a', '')
    text = text.replace('s.a.', '')
    text = text.replace('s.a.u.', '')
    text = text.replace('s.a.d', '')
    text = text.replace('s.l.u.', '')
    text = text.replace('s.p.a.', '')
    text = text.replace('plc', '')
    text = text.replace('a/s', '')
    text = text.replace('args', '')
    text = text.replace('-ag', '')
    text = text.replace('e.v.', '')
    text = text.replace('e. v.', '')
    text = text.replace('e.k.', '')
    text = text.replace('b.v.', '')
    text = text.replace('llp', '')
    text = text.replace(' a.ş.', '')
    text = text.replace('gmbh & co.kg', '')
    text = text.replace('gmbh & co. kg', '')
    text = text.replace('gmbh', '')
    text = text.replace('gesellschaft mbh', '')
    text = text.replace('aktiebolag', '')
    text = text.replace('sp. z o.o.', '')
    text = text.replace('sp.z o.o.', '')
    text = text.replace(' ab', '')
    text = text.replace(' ag', '')
    text = text.replace(' co', '')
    text = text.replace(' bv', '')
    text = text.replace(' p j s c', '')
    text = text.replace(' uk', '')
    text = text.replace(' kg', '')
    text = text.replace(' nv', '')
    text = text.replace(' n.v.', '')
    text = text.replace(' sa/nv', '')
    text = text.replace(' sa', '')
    text = text.replace(' spa', '')
    text = text.replace(' groupe', '')
    text = text.replace(' group', '')
    text = text.replace('groupe ', '')
    text = text.replace(' amba', '')
    text = text.replace(' inc', '')
    text = text.replace(' xet', '')
    text = text.replace(' international', '')
    # Drop substrings enclosed in parentheses
    text = re.sub(r'\([^)]*\)', '', text)
    # Replace non-alphanumeric characters and underscores with a single space
    text = re.sub('[\W_]+', ' ', text, flags=re.UNICODE)
    # Remove leading and trailing spaces
    text = text.strip()

    return text

# Standardize company names. This functions is more general than the other two.
def standardize_company_names(name):
    """
    Standardizes the given name by removing patterns, replacing special characters, converting to lowercase,
    removing leading and trailing spaces, and replacing multiple spaces with a single space.

    Args:
        company_name (str): The corporate names to be standardized.

    Returns:
        str: The standardized name.
    """

    if not isinstance(name, str):
        name = str(name)

    # define patterns to remove
    patterns_to_remove = [
        'co inc', 'incorporated', 'inc', 'co ltd', 'ltd', 'co', 'limited', 'corp',
        'sa', 'plc', 'a/s', 'args', 'group', 'ab', 'amba', 'ag',
        's a', 'b a', 'international', 'holdings', 'holding', 'company'
    ]

    # define special characters to replace with spaces
    special_characters = {
        ',': ' ',
        '.': ' ',
        '/': ' ',
        '&': ' '
    }

    # Convert to lowercase
    standardized_name = name.lower()
    # Replace special characters
    for char, replacement in special_characters.items():
        standardized_name = standardized_name.replace(char, replacement)
    # Remove patterns
    for pattern in patterns_to_remove:
        standardized_name = re.sub(r'\b' + re.escape(pattern) + r'\b', ' ', standardized_name)

    # Replace multiple spaces with single space
    standardized_name = re.sub(r'\s+', ' ', standardized_name)

    # Remove leading and trailing spaces
    standardized_name = standardized_name.strip()

    return standardized_name


# Small function to clean variable names
def clean_variable_names(df):
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(' ', '_')
    return df




#### HELPER FUNCTIONS FOR MERGING ASSET LEVEL DATA ####
'''
Two functions used in prep_asset_level_merge
'''


# FUNCTION to map on different identifiers (e.g., LEI, PERMID)
def map_on_identifiers(df_asset_data, df_portfolio, asset_id_col, df_portfolio_id_col):
    """
    Maps IDs from df_sfi to df_master and returns a dataframe with the mappings.

    Parameters:
    df_asset_data (DataFrame): e.g., df_sfi
    df_portfolio (DataFrame): the main portfolio data
    asset_id_col (str): Column name in asset-data to use for mapping.
    df_portfolio_id_col (str): Corresponding column name in df_portfolio for the mapping.

    Returns:
    DataFrame: A dataframe with 'uid' from df_asset_data and 'permid' from df_portfolio.
    """

    # Filter out rows where the mapping ID is missing in both dataframes
    df_asset_data_filtered = df_asset_data[df_asset_data[asset_id_col].notna()]
    df_portfolio_filtered = df_portfolio[df_portfolio[df_portfolio_id_col].notna()]

    # Perform the mapping using pandas merge
    df_mapped = pd.merge(
        df_asset_data_filtered,
        df_portfolio_filtered,
        left_on=asset_id_col,
        right_on=df_portfolio_id_col,
        how="left"
    )

    # Keep only 'uid' & 'isin', and rename 'isin' column
    new_col_name = "matched_permid_" + asset_id_col
    df_mapped = df_mapped[["uid", "permid"]]
    df_mapped.rename(columns={"permid": new_col_name}, inplace=True)

    return df_mapped


# FUNCTION to consolidate the different matches and print it for better quality control

def merge_and_count_matches(df, columns_to_merge):
    """
        Merges the specified columns in a DataFrame and counts the number of matches in each column.

        Args:
            df (pandas.DataFrame): The DataFrame to perform the merge and count on.
            columns_to_merge (list): A list of column names to merge.

        Returns:
            pandas.DataFrame: The modified DataFrame after merging and removing the original columns.
    """
    # Count matches in individual columns
    matches_count = {col: df[col].notnull().sum() for col in columns_to_merge}

    # Combine columns
    df["matched_permid"] = df[columns_to_merge].bfill(axis=1).iloc[:, 0]
    consolidated_matches = df["matched_permid"].notnull().sum()

    # Delete the original columns
    df.drop(columns=columns_to_merge, inplace=True)

    # Print the results
    for col, count in matches_count.items():
        print(f"Number of matches in '{col}': {count}")
    print(f"Total number of matches in consolidated 'matched_permid': {consolidated_matches}")

    return df


#### HELPER FUNCTIONS TO TURN DT1 DATA INTO A SCORING SYSTEM ####
'''
Both functions are used in apply_decision_tree1.py
'''

def dt1_scoring_log_transform_and_normalize(series):
    """
    Log-transform and normalize a series, handling zeros appropriately.

    Args:
        series (Series): Series to be transformed and normalized.

    Returns:
        Series: Transformed and normalized series.
    """
    # Replace NaNs with zeros
    series = series.fillna(0)

    non_zero = series > 0
    log_transformed = np.log1p(series)  # Use np.log1p to handle zeros properly

    if (non_zero.sum() == 0):
        # If all values are zero, return the original series
        return series

    # Normalize the log_transformed non-zero values
    min_val = log_transformed[non_zero].min()
    max_val = log_transformed[non_zero].max()

    if (min_val == max_val):
        # If all non-zero values are the same, return series of zeros and ones
        normalized = pd.Series(0, index=series.index)
        normalized[non_zero] = 1
    else:
        normalized = pd.Series(0, index=series.index)
        normalized[non_zero] = (log_transformed[non_zero] - min_val) / (max_val - min_val)
        # change any zero values in normalized[non_zero] to half of the min value in normalized[non_zero]
        # Sort the series
        sorted_series = normalized[non_zero].sort_values()
        # Get the unique values to avoid duplicates
        unique_values = sorted_series.unique()
        # Retrieve the second smallest value
        second_smallest_value = unique_values[1]
        # Now assign the lowest value in the non-zero series the second smallest normalized value divided by 2
        normalized[non_zero] = normalized[non_zero].replace(0, second_smallest_value / 2)

    return normalized

def dt1_scoring_calculate_weighted_score(df, weight_dict):
    """
    Calculate the weighted score for a DataFrame based on a given weight dictionary.

    Args:
        df (DataFrame): DataFrame containing the data.
        weight_dict (dict): Dictionary of weights.

    Returns:
        float: Weighted score.
    """
    for key in weight_dict:
        if (key not in df):
            df[key] = 0
    weighted_score = sum(df[key] * weight for key, weight in weight_dict.items())
    return weighted_score


def clean_df_portfolio(df_portfolio):
    """
    Cleans df_portoflio. Can be expanded / adjusted as needed.

    Parameters:
    - df_portfolio (pandas.DataFrame):

    Returns:
    - df_portfolio (pandas.DataFrame): The cleaned DataFrame.
    """

    # there is a werid .0 at the end of the trbc_code_lev3 column
    df_portfolio['trbc_code_lev3'] = df_portfolio['trbc_code_lev3'].str.replace('.0', '')

    # no country iso code for HK companies -> replace "nan" with HK
    df_portfolio['country_iso'] = df_portfolio['country_iso'].replace('nan', 'HK')

    # replace country_iso "JE" with "GB" (Jersey is part of the UK)
    df_portfolio['country_iso'] = df_portfolio['country_iso'].replace('JE', 'GB')

    print("Dataset has been cleaned, ready for further processing.")

    return df_portfolio



##### HELPER FUNCTIONS THAT ARE NOT USED IN THE DEFAULT VERSION OF THIS REPOSITORY #####
'''
Functions that are not used in this public repository but could be useful 
in indicidual circumstances.

Overview:
- get_nace_details: extracts NACE macro sector and sector description at 2-digit level
- city2lonlat: turns city names into longitudal and latitudal data
- fill_missing_location_values_cleaned_df: fills missing location values in cleaned dataframes
'''


## NACE DETAILS
def get_nace_details(df, nace_code_column):
    """
    Description:
    A simple helper script, generating a function that can be applied to the NACE code to generate the macro sector

    Outline of code:
    1) Extract Nace Macro Sector
    2) Extract Nace Sector Description at 2-digit level
    """
    # Extract the first two digits of nace_code and turn it into a numeric variable
    df["nace_temp"] = df[nace_code_column].str[:2]
    df["nace_temp"] = df["nace_temp"].astype(float)

    # 1) Extract Nace Macro Sector

    # Create a new variable nace_macro_sec and define nace macro sectors
    df['nace_macro_sec'] = ""

    # define nace macro sectors (building on old Stata script)
    df['nace_macro_sec'] = np.where((df['nace_temp'] >= 1) & (
            df['nace_temp'] <= 3), "A - Agriculture & Forestry", df['nace_macro_sec'])
    df['nace_macro_sec'] = np.where((df['nace_temp'] >= 5) & (
            df['nace_temp'] <= 9), "B - Mining & quarrying", df['nace_macro_sec'])
    df['nace_macro_sec'] = np.where((df['nace_temp'] >= 10) & (
            df['nace_temp'] <= 33), "C - Manufacturing", df['nace_macro_sec'])
    df['nace_macro_sec'] = np.where((df['nace_temp'] >= 35) & (
            df['nace_temp'] <= 35), "D - Electricity", df['nace_macro_sec'])
    df['nace_macro_sec'] = np.where((df['nace_temp'] >= 36) & (
            df['nace_temp'] <= 39), "E - Water", df['nace_macro_sec'])
    df['nace_macro_sec'] = np.where((df['nace_temp'] >= 41) & (
            df['nace_temp'] <= 43), "F - Construction", df['nace_macro_sec'])
    df['nace_macro_sec'] = np.where((df['nace_temp'] >= 45) & (
            df['nace_temp'] <= 47), "G - Wholesale, Repair of motor vehicles", df['nace_macro_sec'])
    df['nace_macro_sec'] = np.where((df['nace_temp'] >= 49) & (
            df['nace_temp'] <= 53), "H - Transportation and storage", df['nace_macro_sec'])
    df['nace_macro_sec'] = np.where((df['nace_temp'] >= 55) & (
            df['nace_temp'] <= 56), "I - Accommodation & Food Services", df['nace_macro_sec'])
    df['nace_macro_sec'] = np.where((df['nace_temp'] >= 58) & (
            df['nace_temp'] <= 63), "J - Information and Communication", df['nace_macro_sec'])
    df['nace_macro_sec'] = np.where((df['nace_temp'] >= 64) & (
            df['nace_temp'] <= 66), "K - Financial and insurance", df['nace_macro_sec'])
    df['nace_macro_sec'] = np.where((df['nace_temp'] >= 68) & (
            df['nace_temp'] <= 68), "L - Real Estate", df['nace_macro_sec'])
    df['nace_macro_sec'] = np.where((df['nace_temp'] >= 69) & (
            df['nace_temp'] <= 75), "M - Professional, scientific & technical activities", df['nace_macro_sec'])
    df['nace_macro_sec'] = np.where((df['nace_temp'] >= 77) & (
            df['nace_temp'] <= 82), "N - Administrative & support service act", df['nace_macro_sec'])
    df['nace_macro_sec'] = np.where((df['nace_temp'] >= 84) & (
            df['nace_temp'] <= 84), "O - Public Administration and defense", df['nace_macro_sec'])
    df['nace_macro_sec'] = np.where((df['nace_temp'] >= 85) & (
            df['nace_temp'] <= 85), "P - Education", df['nace_macro_sec'])
    df['nace_macro_sec'] = np.where((df['nace_temp'] >= 86) & (
            df['nace_temp'] <= 88), "Q - Human Health and social work activities", df['nace_macro_sec'])
    df['nace_macro_sec'] = np.where((df['nace_temp'] >= 90) & (
            df['nace_temp'] <= 93), "R - Arts, Entertainment and recreation", df['nace_macro_sec'])
    df['nace_macro_sec'] = np.where((df['nace_temp'] >= 94) & (
            df['nace_temp'] <= 96), "S - Other service activities", df['nace_macro_sec'])
    df['nace_macro_sec'] = np.where((df['nace_temp'] >= 97) & (
            df['nace_temp'] <= 98), "T - Activities as households as employers", df['nace_macro_sec'])
    df['nace_macro_sec'] = np.where(
        df['nace_temp'] == 99, "U - Activities of extraterritorial organisations and bodies", df['nace_macro_sec'])
    df['nace_macro_sec'] = np.where(
        df['nace_temp'].isna(), "NA", df['nace_macro_sec'])

    # 2) Extract Nace Sector Description at 2-digit level

    # Import nace code description from .xlsx file
    nace_2digit_desc = pd.read_excel(os.path.join(
        PATH_TO_INPUT_FOLDER, 'nace_level2_description.xlsx'), sheet_name="lvl2_overview",
        dtype={"nace2dig": str})

    # Extract the first two digits of nace_code
    df["nace2dig"] = df[nace_code_column].str[:2]

    # Attach nace description to df
    df = pd.merge(df, nace_2digit_desc, how="left",
                  left_on="nace2dig", right_on="nace2dig")
    del df["nace2dig"]

    # Clean up temporary column
    del df['nace_temp']

    return df


## Function to turn city names into longitudal and latitudal data
"""
Description:
    If only country & city is know, add coordinates (longitude & latitude) to the data.

USER INPUT REQUIRED:
- download & incorporate data from: 
--> https://public.opendatasoft.com/explore/dataset/geonames-all-cities-with-a-population-500/table/?disjunctive.country
--> https://simplemaps.com/data/world-cities
"""

# PLACEHOLDER FUNCTION, adjust as needed
def city2lonlat(df_input, path_to_opendatasoft, path_to_worldcities, googleAPIused):
    """
    This function performs the following tasks:
    1. Reads supplementary city data from OpenDataSoft and WorldCities sources.
    2. Standardizes city names to ensure consistency.
    3. Merges the supplementary data with the input DataFrame based on city names and country codes.
    4. Adds longitude and latitude information to the input DataFrame.
    5. Outputs the enhanced DataFrame with added geographical coordinates.


    NOTES:
    - Fuzzy string matching could increase the number of matches in some cases (e.g, "Milan" vs. "Milano")
    - Check consistency of country codes
    Enhance input city data with longitude and latitude coordinates.

    Args:
    df_input (DataFrame): The input DataFrame with city and country information.
    path_to_opendatasoft (str): File path to the OpenDataSoft data source.
    path_to_worldcities (str): File path to the WorldCities data source.
    googleAPIused (bool): Indicator to use Google API data for missing coordinates.

    Returns:
    DataFrame: The input DataFrame enhanced with 'latitude' and 'longitude' columns.
    """

    ## Ensure consistency of input data

    df_input = df_input.rename(columns={  # ADJUST
        "city_internat": "city_name",
        "ctryiso": "country_iso"
    })

    # Clean data: remove duplicates and missing values (for efficiency)
    df_input_red = df_input.drop_duplicates(subset=["city_name", "country"])
    df_input_red = df_input_red.dropna(subset=["city_name"])

    ## Load OpenDataSoft city data
    df_opendatasoft = pd.read_excel(path_to_opendatasoft, sheet_name="data")
    df_opendatasoft.columns = df_opendatasoft.columns.str.lower().str.replace(" ", "_")
    df_opendatasoft['ascii_name'] = df_opendatasoft['ascii_name'].fillna(
        df_opendatasoft['name'])  # there are 17 missing ascii names
    df_opendatasoft.rename(columns={"ascii_name": "city_name"}, inplace=True)
    df_opendatasoft.drop(columns=['name'], inplace=True)  # not needed anymore
    df_opendatasoft.loc[df_opendatasoft['country_code_2'].notnull(), 'country_code'] = df_opendatasoft[
        'country_code_2']  # e.g., replace Faroe Islands with Denmark
    del df_opendatasoft['country_code_2']
    df_opendatasoft['country_code'] = df_opendatasoft['country_code'].str.split(',').str[
        0]  # for the few cases where there are multiple country codes

    ## Load WorldCities data
    df_worldcities = pd.read_excel(path_to_worldcities, sheet_name="data")
    df_worldcities.columns = df_worldcities.columns.str.lower().str.replace(" ", "_")
    df_worldcities.rename(columns={"city_ascii": "city_name"}, inplace=True)
    df_worldcities.drop(columns=['city'], inplace=True)

    ## Standardize city names
    def standardize_names(data, column):
        data[column] = data[column].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
        data[column] = data[column].str.lower().str.strip()
        return data

    df_input_red["city_name_standardized"] = standardize_names(df_input_red, 'city_name')['city_name']
    df_opendatasoft = standardize_names(df_opendatasoft, 'city_name')
    df_worldcities = standardize_names(df_worldcities, 'city_name')

    # Drop duplicates if there are any
    df_opendatasoft = df_opendatasoft.drop_duplicates(subset=['city_name', 'country_code'])
    df_worldcities = df_worldcities.drop_duplicates(subset=['city_name', 'country_iso2'])

    ## Merge data from OpenDataSoft and WorldCities
    df_opendatasoft['coordinates1'] = df_opendatasoft['latitude'].astype(str) + ", " + df_opendatasoft[
        'longitude'].astype(str)
    df_worldcities['coordinates2'] = df_worldcities['latitude'].astype(str) + ", " + df_worldcities['longitude'].astype(
        str)
    merged_data = pd.merge(df_input_red, df_opendatasoft[['city_name', 'country_code', 'coordinates1']],
                           left_on=['city_name', 'country_iso'], right_on=['city_name', 'country_code'], how='left')
    merged_data = pd.merge(merged_data, df_worldcities[['city_name', 'country_iso2', 'coordinates2']],
                           left_on=['city_name', 'country_iso'], right_on=['city_name', 'country_iso2'], how='left')
    merged_data['coordinates_from_city'] = merged_data['coordinates1'].fillna(merged_data['coordinates2'])
    print("Number of missing coordinates after merge:", merged_data['coordinates_from_city'].isnull().sum())
    print("Number of existing coordinates after merge:", merged_data['coordinates_from_city'].notna().sum())

    ## Add coordinates to the input data
    merged_data = merged_data[['city_name', 'country_iso', 'coordinates_from_city']]
    merged_data = merged_data.drop_duplicates()
    df_input['city_name'] = df_input['city_name'].str.lower().str.strip()
    df_output = pd.merge(df_input, merged_data, on=['city_name', 'country_iso'], how='left')

    # Split up the coordinate column into a longitude and latitude column
    df_output[['latitude', 'longitude']] = df_output['coordinates_from_city'].str.split(',', expand=True)

    # Stripping any potential whitespace
    df_output['latitude'] = df_output['latitude'].str.strip()
    df_output['longitude'] = df_output['longitude'].str.strip()

    # Dropping the original column
    df_output.drop(columns='coordinates_from_city', inplace=True)

    # Only keep essential columns
    df_output = df_output[['isin_par', 'bvdid_sub', 'city_name', 'country_iso', 'level', 'latitude', 'longitude']]

    # Store the cities for which we miss the location information
    df_missing_location = df_output[df_output[['latitude', 'longitude']].isna().any(axis=1)]
    df_missing_location = df_missing_location[['city_name', 'country_iso']].drop_duplicates()
    # Drop rows with NA in either 'city_name' or 'country_iso'
    df_missing_location = df_missing_location.dropna(subset=['city_name', 'country_iso'])
    df_missing_location.to_csv(os.path.join(PATH_TO_INPUT_FOLDER, 'orbis/results/missing_city_locations.csv'))

    if googleAPIused:
        google_api_cities = pd.read_csv(os.path.join(PATH_TO_INPUT_FOLDER, 'ADD HERE THE PATH FOR THE API LOCATIONS'))
        df_output = pd.merge(df_output, google_api_cities, on=['city_name', 'country_iso'], how='left')

    # Only keep rows with longitudal and latitudal information
    df_output = df_output.dropna(subset=['latitude', 'longitude'])

    df_output = df_output.drop_duplicates()

    return df_output


## Another way to extract coordinates from city names
## An API key is required

def fill_missing_location_values_cleaned_df(subsidiary_df_path, google_api_key):
    """
    Fill missing geographical coordinates in the subsidiary DataFrame using Google Maps API.

    This function reads a CSV file containing subsidiary information, uses the Google Maps API to
    obtain missing latitude and longitude data for cities, and writes the enhanced DataFrame back
    to a new CSV file.

    Note:
    - The function requires a valid Google API key.
    - Ensure the API key is kept hidden and secure.

    Args:
    subsidiary_df_path (str): File path to the subsidiary CSV file.
    google_api_key (str): Google Maps API key.

    Returns:
    DataFrame: The input DataFrame with filled 'latitude' and 'longitude' columns.
    """

    if not subsidiary_df_path.endswith('.csv'):
        raise ValueError('im doing some simple string operations that will now not work; please adapt')

    df = pd.read_csv(subsidiary_df_path)

    gmaps = googlemaps.Client(key=google_api_key)

    num_filled = 0

    df['latitude'] = None
    df['longitude'] = None

    # Use iterrows to iterate over DataFrame rows
    for index, row in df.iterrows():
        print(index)
        # Apply the function and store the result in the new column
        geocode_result = gmaps.geocode(row['city_name'])
        if len(geocode_result) == 0:
            continue
        else:
            df.at[index, 'latitude'] = str(geocode_result[0]['geometry']['location']['lat'])
            df.at[index, 'longitude'] = str(geocode_result[0]['geometry']['location']['lng'])
            # row['coordinates_from_city']
            num_filled += 1

    print('NUMBER OF ROWS SUCCESSFULLY FILLED: {}'.format(num_filled))

    df.to_csv(subsidiary_df_path[:-4] + '_nans_filled_not_filtered.csv')

    return df
