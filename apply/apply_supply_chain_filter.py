"""
Description:
    This script adds the IO model scores to the input dataframe.

NOTES:
    - One could add winsorizing / extreme outlier handling where the IO scores are aggregated.
"""

import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from filepaths import PATH_TO_INPUT_FOLDER
from generate.generate_supply_chain_filter import generate_supply_chain_deforestation_footprints
from user_input import EXIOBASE3_YEAR, IO_DEFORESTATION_ATTRIBUTION_DATA_YEAR, region_mapping_ISO2_EXIO


def apply_supply_chain_filter(df_portfolio, df_portfolio_country_sector_pairs):
    """
    This function adds the IO model score to the df_portfolio, based on disaggregated data (if availble).

    Args:
        df_portfolio (pd.DataFrame): Main input portfolio data containing at least the columns
                                        ['nace_code', 'country_iso'].
        df_portfolio_country_sector_pairs (pd.DataFrame): Dataframe containing the region, sector pairs associated
                                    with all the portfolio companies.

    Returns:
        (pd.DataFrame): the input dataframe, but now with the sectoral score added under a new column
                        'IO_supply_chain_score'
    """

    #  Store initial columns for print statement
    initial_columns = set(df_portfolio.columns)

    # load or generate the indirect deforestation impact .pkl file
    filter_path = PATH_TO_INPUT_FOLDER / Path(
        'EXIOBASE3/indirect_deforestation_footprints_country_sector_exio' + str(EXIOBASE3_YEAR) + '_attribution' + str(
            IO_DEFORESTATION_ATTRIBUTION_DATA_YEAR) + '.pkl')

    if os.path.exists(filter_path):
        print('loading: indirect deforestation footprints')
        with open(filter_path, 'rb') as f:
            indirect_deforestation_impacts = pickle.load(f)
        print('DONE')
    else:
        print('generating: indirect deforestation footprints')
        print('NB: this will take several minutes! Wait for textual output!')
        indirect_deforestation_impacts = generate_supply_chain_deforestation_footprints(
            EXIOBASE3_YEAR, IO_DEFORESTATION_ATTRIBUTION_DATA_YEAR)

        print('SAVING INDIRECT IMPACTS')
        # could be changed to .csv saving if preferred
        with open(filter_path, 'wb') as f:
            pickle.dump(indirect_deforestation_impacts, f)
        print('DONE')

    indirect_deforestation_impacts = indirect_deforestation_impacts.squeeze()

    # Load sector mapping NACE to EXIOBASE
    print('loading: sector mapping NACE to EXIOBASE')
    df_nace2exio = pd.read_excel(Path(PATH_TO_INPUT_FOLDER) / "classifications_mapping/NACE2full_EXIOBASEp.xlsx")

    sector_mapping_NACE_EXIO = {}
    for index, row in df_nace2exio.iterrows():
        code = str(row['Code'])  # Convert to string to handle leading zeros
        exiobase_columns = [f'EXIOBASE {i}' for i in range(1, 56)]  # Include all EXIOBASE columns from 1 to 30
        exiobase_activities = [row[col] for col in exiobase_columns if pd.notna(row[col])]

        sector_mapping_NACE_EXIO[code] = exiobase_activities

    # Add score per region-sector pair
    IO_supply_chain_scores = []

    for i in range(len(df_portfolio_country_sector_pairs)):
        nace_code = f'{df_portfolio_country_sector_pairs.iloc[i].nace_code}'
        region = df_portfolio_country_sector_pairs.iloc[i].country_iso

        region_exiobase = region_mapping_ISO2_EXIO[region]

        # Check if nace_code is in sector_mapping_NACE_EXIO
        if nace_code in sector_mapping_NACE_EXIO:
            sectors_exiobase = sector_mapping_NACE_EXIO[nace_code]

            region_sector_pairs = [(region_exiobase, sector) for sector in sectors_exiobase]

            region_sector_pair_footprints = []
            for region_sector_pair in region_sector_pairs:
                try:
                    footprint = indirect_deforestation_impacts[region_sector_pair[0]][region_sector_pair[1]]
                    region_sector_pair_footprints.append(footprint)
                # NB. one could check when this key error happens and rewrite such that this is not needed
                except KeyError:
                    # Handle missing data, you can choose to skip or use a default value
                    # For now, we will skip and continue with the loop
                    warnings.warn('NOTE: missing data encountered in the supply chain filter')
                    continue

            if region_sector_pair_footprints:
                average_over_exiobase_sectors = np.array(region_sector_pair_footprints).mean()
                IO_supply_chain_scores.append(average_over_exiobase_sectors)
            else:
                IO_supply_chain_scores.append(None)  # Handle case with no valid footprints
        else:
            IO_supply_chain_scores.append(None)  # Handle case where nace_code is not found

    df_portfolio_country_sector_pairs['IO_supply_chain_score'] = IO_supply_chain_scores

    # Create score at isin level
    df_portfolio_country_sector_pairs_io_model = df_portfolio_country_sector_pairs.groupby('identifier').apply(
        lambda group: (group['weight_final'] * group['IO_supply_chain_score']).sum()).reset_index()

    # Rename columns
    df_portfolio_country_sector_pairs_io_model.columns = ['identifier', 'io_supply_chain_score']

    # Add disagg IO_supply_chain_score to index_weights
    df_portfolio = df_portfolio.merge(df_portfolio_country_sector_pairs_io_model, on='identifier', how='left')

    # Fill NAs with zeros
    df_portfolio['io_supply_chain_score'] = df_portfolio['io_supply_chain_score'].fillna(0)

    # Print statement for quality control
    new_columns = set(df_portfolio.columns) - initial_columns
    print(f"Number of columns added: {len(new_columns)}")

    print('Distribution of IO scores:')
    print(df_portfolio['io_supply_chain_score'].describe())

    return df_portfolio
