"""
Description:
    This script is designed to collect and preprocess climate trace data.
    
Update:
    Last updated in November 2023

Output:
    A cleaned and structured DataFrame
"""

import os

import pandas as pd

from filepaths import PATH_TO_INPUT_FOLDER, PATH_CLIMATE_TRACE_IDENTIFIERS, \
    PATH_CLIMATE_TRACE_IDENTIFIERS_SOURCE


def process_and_save_climate_trace_data(ownership_threshold=10):
    """
    Process and save Climate Trace data from the specified input folder to the output folder.

    Args:
        ownership_threshold (int): The minimum percent interest to include an asset with multiple owners.

    Returns:
        pd.DataFrame: A DataFrame containing climate trace data.
    """

    ### STEP 0: Prepare LEI and PERMID columns
    df_clt_identifier = pd.read_csv(PATH_CLIMATE_TRACE_IDENTIFIERS)
    df_clt_identifier['org_id'] = df_clt_identifier['org_id'].astype(int)  # turn org_id into integer

    df_clt_identifier_source = pd.read_csv(PATH_CLIMATE_TRACE_IDENTIFIERS_SOURCE)
    df_clt_identifier_source.rename(columns={'name': 'identifier_type'}, inplace=True)

    # attach column "identifier_type" from df_clt_identifier_source to df_clt_identifier via org_id
    df_clt_identifier = pd.merge(df_clt_identifier, df_clt_identifier_source[["org_id", "identifier_type"]], how='left',
                                 on='org_id')

    # rename entries in identifier_type
    rename_dict = {
        "Global Legal Entity Identifier Index": "identifier_lei",
        "S&P Capital IQ": "identifier_sp_cap_iq",
        "PermID: Refinitiv Permanent Identifier": "identifier_permid",
        "Unified Social Credit Identifier": "identifier_usci",
        "UK Companies House": "identifier_uk_ch",
        "US-EIA": "identifier_us_eia",
        "Market Identifier Codes (MIC) ISO 10383 Codes for exchanges and market identification": "identifier_mic"
    }

    # rename entries in identifier_type
    df_clt_identifier['identifier_type'] = df_clt_identifier['identifier_type'].replace(rename_dict)

    df_clt_entity2identifier = df_clt_identifier.pivot_table(
        index='entity_id',
        columns='identifier_type',
        values='value',
        aggfunc='first'
    ).reset_index()

    del df_clt_identifier, df_clt_identifier_source
    df_clt_entity2identifier = df_clt_entity2identifier[
        ['entity_id', 'identifier_lei', 'identifier_permid']]  # most relevant identifiers

    ### STEP 1: LOAD CLIMATE TRACE ASSET DATA
    climate_trace_input_folder = os.path.join(PATH_TO_INPUT_FOLDER, "asset_level_data/climate_trace")

    # Collect all filenames in the input folder ending with _ownership.csv
    climate_trace_files = [file for file in os.listdir(climate_trace_input_folder) if file.endswith("_ownership.csv")]
    temp_list = [pd.read_csv(os.path.join(climate_trace_input_folder, file)) for file in climate_trace_files]
    df_climate_trace = pd.concat(temp_list, ignore_index=True)  # Turning the list of dataframes into a single dataframe
    del temp_list

    ### DATA CLEANING
    # Store assets with one owner in separate df
    unique_assets = df_climate_trace.drop_duplicates(subset='source_id', keep=False)

    # Store assets with multiple owners in separate df (and keep only >10% ownership)
    multiple_owners = df_climate_trace[df_climate_trace.duplicated(subset='source_id', keep=False)]
    multiple_owners = multiple_owners[multiple_owners['percent_interest_parent'] >= ownership_threshold]

    # Combine both
    df_climate_trace = pd.concat([unique_assets, multiple_owners], ignore_index=True)

    # add identifier columns to ultimate_parent_id, and company_id, and source_id
    df_climate_trace = pd.merge(df_climate_trace, df_clt_entity2identifier, how='left', left_on='ultimate_parent_id',
                                right_on='entity_id')
    df_climate_trace.rename(columns={'identifier_lei': 'parent_lei',
                                     'identifier_permid': 'parent_permid'}, inplace=True)
    df_climate_trace.drop(columns='entity_id', inplace=True)  # remove entity_id column

    df_climate_trace = pd.merge(df_climate_trace, df_clt_entity2identifier, how='left', left_on='company_id',
                                right_on='entity_id')
    df_climate_trace.rename(columns={'identifier_lei': 'company_lei',
                                     'identifier_permid': 'company_permid'}, inplace=True)
    df_climate_trace.drop(columns='entity_id', inplace=True)  # remove entity_id column

    df_climate_trace = pd.merge(df_climate_trace, df_clt_entity2identifier, how='left', left_on='source_id',
                                right_on='entity_id')
    df_climate_trace.rename(columns={'identifier_lei': 'operator_lei',
                                     'identifier_permid': 'operator_permid'}, inplace=True)
    df_climate_trace.drop(columns='entity_id', inplace=True)  # remove entity_id column

    ### PREPARE OUTPUT

    # Clean up sector
    df_climate_trace['sector'] = df_climate_trace['original_inventory_sector'].str.replace("-", " ")
    relevant_columns = ['source_id', 'source_name', 'company_name', 'ultimate_parent_name',
                        'parent_lei', 'parent_permid', 'operator_lei', 'operator_permid',
                        'company_lei', 'company_permid', 'iso3_country', 'sector', 'lat', 'lon',
                        'percent_interest_parent']

    df_climate_trace = df_climate_trace[relevant_columns]
    df_climate_trace.rename(columns={'source_id': 'asset_id',
                                     'source_name': 'asset_name',
                                     'company_name': 'company_name',
                                     'ultimate_parent_name': 'parent_name',
                                     'iso3_country': 'country',
                                     'lat': 'latitude',
                                     'lon': 'longitude'}, inplace=True)
    df_climate_trace.reset_index(drop=True, inplace=True)

    ## CLEANING

    # turn asset_name, company_name, parent_name into strings
    string_variables = ['asset_name', 'company_name', 'parent_name']
    for var in string_variables:
        df_climate_trace[var] = df_climate_trace[var].astype(str)
        df_climate_trace[var].fillna('', inplace=True)  # fill nan with empty string

    # create unique id for each row, counting from 0 to len(df_climate_trace)
    df_climate_trace['uid'] = ['CLT_' + str(num) for num in list(range(len(df_climate_trace)))]

    return df_climate_trace
