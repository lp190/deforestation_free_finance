"""
Description:
    This script integrates deforestation hotspot scores into a given dataframe. 
    Depending on the type of asset specified, it processes either asset-level or subsidiary-level data. 
    The deforestation hotspots are identified based on a specified distance threshold. (see user_input file)

Output:
    The script returns the input dataframe with the deforestation hotspot scores added as new columns.
"""

import os

import pandas as pd

from filepaths import PATH_TO_OUTPUT_FOLDER, PATH_TO_DEFORESTATION_HOTSPOTS_2023, \
    PATH_TO_DISAGGREGATED_ASSETS, PATH_TO_SUBSIDIARY_LOCATIONS
from prep.prep_hotspot import prep_gfw_hotspot
from user_input import IMPACT_DICT


def apply_deforestation_hotspots_assets(df_portfolio, type_of_asset, distance_threshold):
    """
    Applies deforestation hotspots to the given index weights.

    Args:
        df_portfolio (pd.DataFrame): The main portfolio data.
        type_of_asset (string): either 'asset' or 'subsidiary' is currently supported. otherwise ValueError
        distance_threshold (float): The distance threshold for identifying hotspots.

    Returns:
        pd.DataFrame: The index weights dataframe with hotspots applied.
    """

    if type_of_asset == 'asset':
        output_path = os.path.join(PATH_TO_OUTPUT_FOLDER, 'internal_data/hotspot_scores_assets.csv')
    elif type_of_asset == 'subsidiary':
        output_path = os.path.join(PATH_TO_OUTPUT_FOLDER, 'internal_data/hotspot_scores_subsidiary.csv')
    else:
        raise ValueError('not yet supported')

    if os.path.exists(output_path):
        print('loading: hotspot_scores')
        hotspot_scores = pd.read_csv(output_path)
        print('DONE')
    else:
        print('preparing: GFW hotspot scores, for {}'.format(type_of_asset))
        hotspot_scores = prep_gfw_hotspot(df_portfolio, distance_threshold, type_of_asset,
                                          IMPACT_DICT, PATH_TO_DEFORESTATION_HOTSPOTS_2023,
                                          PATH_TO_DISAGGREGATED_ASSETS, PATH_TO_SUBSIDIARY_LOCATIONS)

        print('saving the scores')
        hotspot_scores.to_csv(output_path)
        print('DONE')

    # Store original columns for comparison
    original_columns = set(df_portfolio.columns)

    # TEMP FIX
    # --------------------------------
    # Before merge, turn 'identifier' into a string
    hotspot_scores['identifier'] = hotspot_scores['identifier'].astype(str)

    # Remove column "Unnamed: 0" if it exists
    if 'Unnamed: 0' in hotspot_scores.columns:
        hotspot_scores.drop(columns=['Unnamed: 0'], inplace=True)

    # ------------------------- end of TEMP FIX

    # Merge the hotspot data with the input dataframe
    df_portfolio = pd.merge(df_portfolio, hotspot_scores, left_on=['identifier'],
                            right_on=['identifier'], how='left')

    # Print which columns have been added
    new_columns = set(df_portfolio.columns) - original_columns
    print(f"New columns added: {', '.join(new_columns)}")

    for column in new_columns:
        non_zero_count = (df_portfolio[column] != 0).sum()
        print(f"Number of non-zero values in '{column}': {non_zero_count}")

    return df_portfolio
