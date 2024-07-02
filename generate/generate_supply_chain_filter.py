"""
Filename: generate_supply_chain_deforestation_footprints.py

Description:
    This script derives indirect deforestation footprints by processing EXIOBASE data and calculating the Leontief
    inverse to account for indirect effects in supply chains.

Update:
    Last updated in June 2024

Sources:
    - Code from Richard Wood: https://github.com/rich-wood/exiobase_luc/blob/main/process_luc.py
    - UPDATED attribution data: https://zenodo.org/records/10633818


NOTES/:
    - NB. one could check for small values in y; set those to 0 in M? or scale down? These values could be inaccurate.
    - Improve readability & documentation
"""

import os

import pandas as pd
import pymrio

from filepaths import PATH_TO_INPUT_FOLDER


def generate_supply_chain_deforestation_footprints(exiobase_year, io_deforestation_attribution_data_year):
    """
    NOTE. This function has been written with code that has been adapted from:
    https://github.com/rich-wood/exiobase_luc/blob/main/process_luc.py; and then adapted to use the new attribution data

    It generates a pd.Series with indirect/downstream deforestation footprint per unit output for each (region, sector)
    pair.

    It processes EXIOBASE data, calculates the Leontief inverse to account for indirect effects, and computes
    the deforestation footprint for each sector-region pair. The function is called by 'apply_supply_chain_filter'
    if the series has not been previously created.

    Args:
        exiobase_year (int): year of the Exiobase model
        io_deforestation_attribution_data_year (int): year of the attribution data that will be loaded (note: this does
                                                        have to correspond to the Exiobase year, but be aware that
                                                        this means that the trade stats will be of a different year
                                                        than the deforestation attribution data)

    Returns:
        pd.Series: A pandas series with the indirect deforestation footprint per unit output for each (region, sector) pair.
    """

    exio3_folder_path = os.path.join(PATH_TO_INPUT_FOLDER, 'EXIOBASE3')

    # FROM R. WOOD. there is a idscrepancy in codes vs names in some files, so load the full classification for renaming later
    EX2i = pd.read_csv(os.path.join(exio3_folder_path, 'EXIOBASE20i.txt'), index_col=0, usecols=[1, 3], header=None,
                       sep='\t')
    EX2i_dict = dict(EX2i.iloc[:, -1])

    # load deforestation data // NOTE: this is ONLY used to re-index the actual deforestation data appropriately
    df2018 = pd.read_csv(os.path.join(
        exio3_folder_path, "OutputExiobaseStructured.csv"), header=0, index_col=[0, 1], usecols=[0, 1, 3, 4, 5])

    df_allyears = pd.read_csv(os.path.join(
        exio3_folder_path, "UpdatedAttributionData.csv"), header=0, index_col=[0, 1])

    # rename allyears to full names for consistency with exio3
    df_allyears.rename(EX2i_dict, axis=0, inplace=True)

    # if the path to exiobase3 does not exist we need to download exiobase3
    exio3_folder_model_path = os.path.join(exio3_folder_path, 'EXIOBASE_3_8_2')
    if not os.path.exists(exio3_folder_model_path):
        print("DOWNLOADING EXIOBASE3")
        print("This will take some time! Wait for textual output!")
        _ = pymrio.download_exiobase3(storage_folder=exio3_folder_model_path, system="ixi", years=[exiobase_year])
        print("DONE")
    exio3 = pymrio.parse_exiobase3(path=exio3_folder_model_path + '/IOT_' + str(exiobase_year) + '_ixi.zip')
    print("PARSED EXIOBASE3")

    # calculate the Leontief inverse (input requirements associated with one unit of output)
    # For that we use the A matrix (=inter-industry coefficients (direct requirements matrix) // i.e.
    # internal consumption matrix). L = inverse matrix of (I - A) as well as x; the outputs for each sector
    L = pymrio.calc_L(exio3.A)
    x = pymrio.calc_x_from_L(L, exio3.Y.sum(axis=1))

    df_yr = df_allyears.loc[df_allyears['Year'] == io_deforestation_attribution_data_year]
    # reindex (and transpose) so that matrix multiplication can be later performed
    # + pick out only deforestation area in hectares, since that is what we care about
    # NB!! if you do want to add the emissions, you'll have to adapt the underlying attribution data file
    df_yr_full = df_yr.reindex_like(df2018).fillna(0).T.drop(
        ['Deforestation_emissions_incl_peat_MtCO2', 'Deforestation_emissions_excl_peat_MtCO2'])

    # calculate intensities (notation: s):: nb. this is merely a computation of the normalized flows
    s = pymrio.calc_S(df_yr_full, x)  # nb. merely calls calc_A under the hood (direct requirement matrix)

    # S already has the type of units we care about, namely hectares per unit output, in order to 'translate units
    # of output to units of input' we multiply with L (I-A)-1 = I+A+A2+A3+... to account for the indirect effects

    # NB: one could check for small values in y; set those to 0 in M? or scale down? those values could be inaccurate
    footprint_downstream_per_unit_output = pymrio.calc_M(s, L).T

    print('DONE COMPUTING INDIRECT DEFORESTATION FOOTPRINT')

    return footprint_downstream_per_unit_output
