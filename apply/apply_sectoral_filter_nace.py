"""
Description:
    Adds sectoral flags based on NACE codes to the input dataframe.
"""

from pathlib import Path

# import packages
import pandas as pd

from filepaths import PATH_TO_OUTPUT_FOLDER, PATH_TO_SECTOR_FLAGS, PATH_TO_NACE_GICS_MAPPING
from generate.generate_sectoral_filter_nace import generate_sectoral_nace_flags


def apply_sectoral_filters_nace(df_portfolio, df_portfolio_country_sector_pairs):
    """
    This function applies the sectoral filter(s) and adds the score(s) as new columns to the input dataframe.

    Args:
        df_portfolio (pd.DataFrame): Main portfolio data. Needs to contain the column 'nace_code'.
        df_portfolio_country_sector_pairs (pd.DataFrame): the disaggregated data

    Returns:
        pd.DataFrame: The portfolio DataFrame with the sectoral scores added.
    """
    sectoral_flags_path = Path(PATH_TO_OUTPUT_FOLDER) / 'internal_data/sectoral_nace_flags.csv'
    dtype_dict = {'nace_code': str}

    # Load or generate the sectoral NACE flags
    if sectoral_flags_path.exists():
        print('Loading: sectoral_nace_flags.csv')
        sectoral_nace_flags = pd.read_csv(sectoral_flags_path, dtype=dtype_dict)
        print('DONE')
    else:
        print('Generating: nace_to_encore_deforestation.csv')
        sectoral_nace_flags = generate_sectoral_nace_flags(PATH_TO_NACE_GICS_MAPPING, PATH_TO_SECTOR_FLAGS)
        sectoral_nace_flags.to_csv(sectoral_flags_path, index=False)
        print('DONE')

    # Merge the sectoral flags to disaggreated data
    df_portfolio_country_sector_pairs = pd.merge(df_portfolio_country_sector_pairs, sectoral_nace_flags, on='nace_code',
                                                 how='left')

    # Group by 'identifier' to perform the calculations on the company level
    print("Grouping by 'identifier' to calculate scores...")
    grouped = df_portfolio_country_sector_pairs.groupby('identifier')

    # Use agg to perform all the calculations
    df_portfolio_country_sector_pairs = grouped.apply(lambda x: pd.Series({
        'flag_direct_score': (x['weight_final'] * x['flag_direct']).sum(),
        'flag_indirect_score': (x['weight_final'] * x['flag_indirect']).sum()
    })).reset_index()

    print("Scores calculated successfully.")

    # Add the three scores to df_portfolio
    print("Merging scores with portfolio data...")
    df_portfolio = df_portfolio.merge(df_portfolio_country_sector_pairs[
                                          ['identifier', 'flag_direct_score', 'flag_indirect_score']],
                                      on='identifier',
                                      how='left')

    # Check for NAs in the two new variables and ill them with zeros if any
    df_portfolio['flag_direct_score'] = df_portfolio['flag_direct_score'].fillna(0)
    df_portfolio['flag_indirect_score'] = df_portfolio['flag_indirect_score'].fillna(0)

    # Print statements for clarity
    print("The two new variables 'flag_direct_score' and 'flag_indirect_score' have been added to the portfolio data.")
    print(
        "The importance of each (country-)sector pair was taken into account when attributing the direct and indirect sector-specific deforestation exposure to each company.")
    print("Scores merged successfully with portfolio data.")

    return df_portfolio
