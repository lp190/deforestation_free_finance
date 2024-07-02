"""
Description:
    This script generates sectoral flags based on NACE codes by transforming GICS-based sectoral flags using a sector
    crosswalk. The script takes a conservative approach by assigning a positive flag if any of the GICS sectors associated
    with the NACE code has a positive flag. This ensures that the flags are not diluted in cases of one-to-many mappings.

Update:
    Last updated in 10/2023

Source:
    This script leverages a NACE-to-GICS mapping received from ENCORE.

Output:
    A DataFrame with columns for NACE code, direct flag, and indirect flag.

NOTES:
- There are some mismatches due to updates in GICS sectors as of May 2023. These are handled manually within the script.
"""

# import packages
import pandas as pd


def generate_sectoral_nace_flags(path_to_nace_gics_mapping, path_to_literature_sectoral):
    """
    This function generates transforms the GICS-based sectoral flags to NACE using a sector crosswalk. It takes the
    conservative approach of giving a positive flag if any of the GICS sectors associated with the NACE code (in
    case of one-many mappings) has a positive flag (rather than, for instance, taking the average). This happens rarely.

    It is called by the 'apply_sectoral_filter_nace' function when this dataframe has not been previously created.

    Args:
        path_to_nace_gics_mapping (str): Path to the NACE-to-GICS mapping data.
        path_to_literature_sectoral (str): Path to sectoral filter data based on literature.

    Returns:
        mapping (pd.DataFrame): columns = nace_code,  flag_direct,  flag_indirect
    """

    # load nace to gics mapping which can be used to match the literature flags
    mapping = pd.read_excel(path_to_nace_gics_mapping)

    mapping.rename(
        columns={'NACE Code ': 'nace_code'},
        inplace=True)

    # load sectoral filter based on the literature (this is on GICS level)
    literature_sectoral = pd.read_excel(path_to_literature_sectoral, sheet_name=1)

    literature_sectoral = literature_sectoral[['lvl4_code', 'flag_direct', 'flag_indirect']]

    # There are some GICS code mismatches in the two files because of the GICS update in May 2023. Let's solve them manually.
    # Most are trivial. The ones that are not:
    # 20304020: 20304040, Trucking: Passenger Ground Transportation
    # 60101060: 60106010, Residential REITs: now mapped to only multi-residential (rather than average of multi and single...)
    # 25502020: 25503030, internet retail: now mapped to only broadline retail (rather than average of all retail types)
    # 60101080: 60108040, specialised REITs: now mapped to Timber REITs to be CONSERVATIVE (rather than average of all specialised REIT)
    # 45102020: 45102010, data to IT services. the literature file seems to just miss the "45102020" sector
    subindustry_change_mapping = {60102010: 60201010,
                                  60102020: 60201020,
                                  60102030: 60201030,
                                  60102040: 60201040,
                                  20304020: 20304040,
                                  30202009: 30202010,
                                  30202011: 30202010,
                                  60101020: 60102510,
                                  25503010: 25503030,
                                  60101030: 60103010,
                                  25503020: 25503030,
                                  60101040: 60104010,
                                  60101050: 60105010,
                                  60101060: 60106010,
                                  25502020: 25503030,
                                  60101070: 60107010,
                                  60101080: 60108040,
                                  45102020: 45102010,
                                  40102010: 40201050}

    mapping['GICS Sub-Industry Code '] = mapping['GICS Sub-Industry Code '].replace(subindustry_change_mapping)

    mapping = pd.merge(
        mapping,
        literature_sectoral,
        left_on=['GICS Sub-Industry Code '],
        right_on=['lvl4_code'],
        how='left')

    # now group by nace code and average the associated scores
    mapping = mapping.groupby('nace_code')[
        ['flag_direct', 'flag_indirect']].mean().reset_index()

    # to be conservative let's set flag_direct and flag_indirect to 1.0 if any of the associated gics codes had a 1.0
    mapping.loc[mapping['flag_direct'] != 0, 'flag_direct'] = 1.0
    mapping.loc[mapping['flag_indirect'] != 0, 'flag_indirect'] = 1.0

    mapping['nace_code'] = mapping['nace_code'].str.replace('_', '')

    return mapping
