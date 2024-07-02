"""
Description:
    This script combines all processed open-source asset-level datasets into one single dataset.
"""

import numpy as np
import pandas as pd

from generate.generate_asset_level_GEM import process_and_save_gem_data
from generate.generate_asset_level_SFI import process_and_save_sfi_data
from generate.generate_asset_level_climate_trace import process_and_save_climate_trace_data


def combine_asset_datasets():
    """
    Combines and processes asset-level datasets from different sources.

    Returns:
    df_combined (pandas.DataFrame): Combined and processed asset-level dataset.
    """

    print("Processing and combining asset-level datasets...")

    # Load all datasetsprint
    print("Processing data from Global Energy Monitor (GEM)...")
    df_gem = process_and_save_gem_data()
    df_gem['data_source'] = 'GEM'

    print("Processing data from Climate Trace...")
    df_clt = process_and_save_climate_trace_data()
    df_clt['data_source'] = 'Climate Trace'
    df_clt.rename(columns={"company_name": "owner_name"}, inplace=True)

    print("Processing data from Spatial Finance Initiative (SFI)...")
    df_sfi = process_and_save_sfi_data()
    df_sfi['data_source'] = 'SFI'

    # Consolidate the different ownership columns
    '''
    Climate Trace
    - parent_name (and LEI and permid)
    - owner_name (and LEI and permid)
    
    SFI
    - parent_name (and LEI and permid)
    - owner_name  (and LEI and permid)
    
    GEM
    - parent_name (and permid)
    - owner (and permid)
    - operator (and permid)
    
    Therefore, let's consolidate as follows:
    - parent_name (and parent_lei and parent_permid)
    - owner_name (and owner_lei and owner_permid)
    - operator_name (and operator_lei and operator_permid)
    '''

    # Rename according to the above
    df_gem.rename(columns={"ownership_parent_id": "parent_permid",
                           "ownership_owner_id": "owner_permid",
                           "ownership_operator_id": "operator_permid"}, inplace=True)

    # Merge them
    df_combined = pd.concat([df_gem, df_clt, df_sfi], axis=0).reset_index(drop=True)

    # Create blank column operator_lei for sake of consistency
    df_combined['operator_lei'] = np.nan

    # Define columns to keep and perform cleaning
    cols_to_keep = ['uid', 'asset_name', 'country', 'start_year', 'latitude', 'longitude',
                    'parent_name', 'parent_permid', 'parent_lei', 'owner_name', 'owner_permid',
                    'owner_lei', 'operator_name', 'operator_permid', 'operator_lei',
                    'capacity', 'capacity_unit', 'sector', 'data_source']

    df_combined = df_combined[cols_to_keep]

    # Cleaning process
    id_cols = ['parent_permid', 'parent_lei', 'owner_permid', 'owner_lei', 'operator_permid', 'operator_lei']
    df_combined[id_cols] = df_combined[id_cols].astype(str)
    df_combined[id_cols] = df_combined[id_cols].apply(lambda x: x.str.replace('.0', ''))  # remove ".0"

    # Replace 'NA' and 'nan' strings with actual NaN values
    df_combined[id_cols] = df_combined[id_cols].replace(['NA', 'nan'], np.nan)

    # Convert numeric columns
    num_cols = ['start_year', 'latitude', 'longitude']
    df_combined[num_cols] = df_combined[num_cols].apply(pd.to_numeric, errors='coerce')

    # Print statement for better tracking in main script
    print("Asset-level datasets processed and combined!")

    return df_combined
