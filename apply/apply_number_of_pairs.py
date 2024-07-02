"""
Filename: apply_number_of_pairs.py

Description:
    This script simply adds the number of location-sector pairs per dataset to the input dataframe.
    Example:    Company A has 3 location-sector pairs based on asset-level data, 
                2 location-sector pairs based on revenue data, 
                and 10 location-sector pair based on Subsidiary data.

"""
# import packages
import pandas as pd

from filepaths import PATH_TO_DISAGGREGATED_ASSETS


def apply_number_of_pairs(df_portfolio, df_portfolio_country_sector_pairs):
    """
    Args:
        df_portfolio: portfolio data
        df_portfolio_country_sector_pairs: country-sector pairs

    Returns:
        df_portfolio: file with added columns, stating the depth of input data
    """

    # Store initial columns for print statement
    initial_columns = df_portfolio.columns

    # Attach number of country-sector pairs that were used
    country_sector_pairs_count = df_portfolio_country_sector_pairs["identifier"].value_counts().reset_index()
    country_sector_pairs_count.columns = ["identifier", "number_of_pairs_country_sector"]
    country_sector_pairs_count["identifier"] = country_sector_pairs_count["identifier"].astype(str)

    # Attach number of disaggregated assets to the portfolio data
    path_to_disaggregated_asset = PATH_TO_DISAGGREGATED_ASSETS
    assets_count = pd.read_csv(path_to_disaggregated_asset, usecols=['permid'])
    assets_count = assets_count['permid'].value_counts().reset_index()
    assets_count.columns = ['identifier', 'number_of_pairs_asset']
    assets_count["identifier"] = assets_count["identifier"].astype(str)

    ### IF MORE DATA SOURCES WERE USED, ADD THEM HERE ###
    # (such as subsidiaries, revenue splits, etc.)

    # Attach number of pairs per dataset to the portfolio data
    df_portfolio = pd.merge(df_portfolio, country_sector_pairs_count, how='left', on='identifier')
    df_portfolio = pd.merge(df_portfolio, assets_count, how='left', on='identifier')

    # Fill NAs with zeros for the number of pairs columns
    df_portfolio["number_of_pairs_asset"].fillna(0, inplace=True)
    df_portfolio["number_of_pairs_country_sector"].fillna(0, inplace=True)

    # Print the columns that were added
    added_columns = df_portfolio.columns.difference(initial_columns)
    print(f"Added columns: {added_columns}")

    return df_portfolio
