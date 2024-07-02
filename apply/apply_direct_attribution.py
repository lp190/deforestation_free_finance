"""
Description:
    This script integrates direct deforestation attribution scores into a given dataframe.
    It processes the data by mapping NACE codes to FAO sectors, loading or generating deforestation footprints,
    and calculating deforestation exposure for each sector-region pair.

Output:
    The updated input dataframe with added direct deforestation attribution scores is returned.
"""

import os
import warnings

import numpy as np
import pandas as pd

from filepaths import PATH_TO_INPUT_FOLDER, PATH_TO_NACE_FAO_MAPPING, PATH_TO_UPDATED_DEFORESTATION_ATTRIBUTION_DATA, \
    PATH_TO_FAOSTAT_2018, PATH_TO_GDP_DATA_2018, PATH_TO_FAOSTAT_PRODUCTION_2018, \
    PATH_TO_FAOSTAT_PRODUCTION_PRICE_2018, PATH_TO_DEDUCE_MAPPING
from generate.generate_direct_attribution_filter import generate_direct_deforestation_footprints
from user_input import DIRECT_ATTRIBUTION_YEAR, DIRECT_ATTRIBUTION_HECTARE_CUTOFF, USE_AMORTIZED


def apply_direct_attribution_filter(df_portfolio, df_portfolio_country_sector_pairs):
    """
    Applies direct deforestation attribution scores to the given portfolio dataframe.

    Args:
        df_portfolio (pd.DataFrame): Main input portfolio data containing at least the columns
                                        ['nace_code', 'country_iso'].
        df_portfolio_country_sector_pairs (pd.DataFrame): df_portfolio merged with all the assets found for the
                                                            portfolio companies.

    Returns:
        pd.DataFrame: The input dataframe with the sectoral score added under a new column 'direct_attribution_score'.
    """

    #  Store initial columns for print statement
    initial_columns = set(df_portfolio.columns)

    # load or generate the indirect deforestation impact .csv file
    filter_path = os.path.join(PATH_TO_INPUT_FOLDER,
                               'direct_deforestation_footprints_country_sector_' + str(
                                   DIRECT_ATTRIBUTION_YEAR) + '.csv')

    mapping = pd.read_excel(PATH_TO_NACE_FAO_MAPPING, sheet_name='nace_to_fao')

    # now, make the mapping usable
    mapping_nace_fao = {}
    for index, row in mapping.iterrows():
        nace = str(row['NACE'])
        fao_columns = [f'fao{i}' for i in range(1, 47)]  # Include all columns from fao1 to fao46
        fao_activities = [int(row[col]) for col in fao_columns if pd.notna(row[col])]
        mapping_nace_fao[nace] = fao_activities

    if os.path.exists(filter_path):
        print('loading: direct deforestation footprints')
        direct_deforestation_impacts = pd.read_csv(filter_path)
        print('DONE')
    else:
        print('generating: direct deforestation footprints')
        direct_deforestation_impacts = generate_direct_deforestation_footprints(
            PATH_TO_UPDATED_DEFORESTATION_ATTRIBUTION_DATA, DIRECT_ATTRIBUTION_YEAR, PATH_TO_FAOSTAT_2018,
            PATH_TO_GDP_DATA_2018, PATH_TO_FAOSTAT_PRODUCTION_2018, PATH_TO_FAOSTAT_PRODUCTION_PRICE_2018,
            PATH_TO_DEDUCE_MAPPING, DIRECT_ATTRIBUTION_HECTARE_CUTOFF, USE_AMORTIZED)

        print('SAVING DIRECT IMPACTS')
        direct_deforestation_impacts.to_csv(filter_path)
        print('DONE')

    # now, generate the (sector, region) pairs for the headquarters

    direct_attribution_scores = []

    for i in range(len(df_portfolio_country_sector_pairs)):
        nace_code = f'{df_portfolio_country_sector_pairs.iloc[i].nace_code}'
        region = df_portfolio_country_sector_pairs.iloc[i].country_iso

        if len(region) == 2:
            iso = 'ISO2'
        elif len(region) == 3:
            iso = 'ISO'
        else:
            direct_attribution_scores.append(None)
            warnings.warn('something is wrong with the region in index weights')
            continue

        # Check if nace_code is in sector_mapping_NACE_EXIO
        if nace_code in mapping_nace_fao:
            sectors_fao = mapping_nace_fao[nace_code]

            region_sector_pairs = [(region, sector) for sector in sectors_fao]

            region_sector_pair_hectares_exposures = []
            region_sector_pair_hectares_sizes = []
            for region_sector_pair in region_sector_pairs:
                try:
                    region_sector_pair_hectares_exposure = direct_deforestation_impacts[
                        (direct_deforestation_impacts[iso] == region_sector_pair[0]) &
                        (direct_deforestation_impacts['FAO'] == region_sector_pair[1])].Deforestation_hectares
                    region_sector_pair_hectares_size = direct_deforestation_impacts[
                        (direct_deforestation_impacts[iso] == region_sector_pair[0]) &
                        (direct_deforestation_impacts['FAO'] == region_sector_pair[1])].Size
                    if len(region_sector_pair_hectares_exposure) > 0:
                        region_sector_pair_hectares_exposures.append(float(region_sector_pair_hectares_exposure))
                        region_sector_pair_hectares_sizes.append(float(region_sector_pair_hectares_size))
                # todo check when this key errors happens and rewrite such that this is not needed
                except KeyError:
                    warnings.warn('NOTE: missing data encountered in the direct attribution filter')
                    # Handle missing data, you can choose to skip or use a default value
                    # For now, we will skip and continue with the loop
                    continue

            if region_sector_pair_hectares_exposures:
                sum_exposure = np.array(region_sector_pair_hectares_exposures).sum()
                sum_size = np.array(region_sector_pair_hectares_sizes).sum()
                direct_attribution_scores.append(sum_exposure / sum_size)
            else:
                direct_attribution_scores.append(None)  # Handle case with no valid footprints
        else:
            direct_attribution_scores.append(None)  # Handle case where nace_code is not found

    df_portfolio_country_sector_pairs['direct_attribution_scores'] = direct_attribution_scores
    df_portfolio_country_sector_pairs['direct_attribution_scores'].fillna(0, inplace=True)

    # Create score at company level
    df_portfolio_country_sector_pairs_pendrill = df_portfolio_country_sector_pairs.groupby('identifier').apply(
        lambda group: (group['weight_final'] * group['direct_attribution_scores']).sum()).reset_index()

    # Rename columns
    df_portfolio_country_sector_pairs_pendrill.columns = ['identifier', 'direct_attribution_score']

    # Add disagg direct attribution score to index_weights
    df_portfolio = df_portfolio.merge(df_portfolio_country_sector_pairs_pendrill, on='identifier', how='left')

    # Fill NAs with zeros
    df_portfolio['direct_attribution_score'] = df_portfolio['direct_attribution_score'].fillna(0)

    # Print statement for quality control
    new_columns = set(df_portfolio.columns) - initial_columns
    print(f"Number of columns added: {len(new_columns)}")

    for column in new_columns:
        non_zero_count = (df_portfolio[column] != 0).sum()
        print(f"Number of non-zero values in '{column}': {non_zero_count}")

    return df_portfolio
