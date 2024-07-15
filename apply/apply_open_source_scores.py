"""
Description: 
    This script accesses all open sources files and merges them with the portfolio data.
    
Outline of the script:
    1) IMPORT DATA from generate folder & merge & clean
    2) COMBINE WITH PORTFOLIO DATA
    
NOTES:
    - Users could add identifiers to identifiers_portfolio to improve matching
                    (currently used: ['permid', 'lei', 'ticker', 'ric'])
    - The positive flags are relatively arbitrary and can be adjusted as needed.
    - NB. user would have to add FAIRR variables themselves (SEE: generate_FAIRR.py)
"""

# import packages
import pandas as pd
import numpy as np

# import paths and functions
from filepaths import PATH_TO_SPOTT_PALM_OIL_QUESTIONS, \
    PATH_TO_SPOTT_TIMBER_ETC_QUESTIONS, PATH_TO_SPOTT_RUBBER_QUESTIONS, PATH_TO_DEFORESTATION_ACTION_TRACKER
from generate.generate_FAIRR import generate_FAIRR
from generate.generate_SBTI import generate_SBTI, variables_of_interest_SBTI
from generate.generate_SPOTT import generate_SPOTT, variables_of_interest_SPOTT, important_questions_palm_oil, \
    important_questions_timber, important_questions_rubber
from generate.generate_TNFD import generate_TNFD
from generate.generate_WBA import generate_WBA
from generate.generate_deforestation_action_tracker import generate_dat_scores
from generate.generate_food_emissions_50 import generate_fe_50, variables_of_interest_FE_50
from user_input import STRING_MATCHING_THRESHOLD_OPEN_SOURCE
from utils import standardize_company_names


### FUNCTION ###

def apply_open_source_scores(df_portfolio,
                             vars_of_interest_FE_50=variables_of_interest_FE_50,
                             vars_of_interest_SBTI=variables_of_interest_SBTI,
                             vars_of_interest_SPOTT=variables_of_interest_SPOTT,
                             string_matching_threshold=STRING_MATCHING_THRESHOLD_OPEN_SOURCE):

    """
    Integrate the open-source scores.

    This function imports various open-source datasets, merges them into a base DataFrame, and then
    integrates this data with the provided portfolio DataFrame. It uses identifier matching and fuzzy string matching
    to combine the datasets and assigns flags based on the presence of relevant scores.

    Args:
        df_portfolio (pd.DataFrame): The portfolio DataFrame to which open-source scores will be added (main df).
        vars_of_interest_FE_50 (tuple): Variables of interest for the FE_50 dataset.
        vars_of_interest_SBTI (tuple): Variables of interest for the SBTI dataset.
        vars_of_interest_SPOTT (tuple): Variables of interest for the SPOTT dataset.
        string_matching_threshold (int): Threshold score for fuzzy string matching.

    Returns:
        pd.DataFrame: The portfolio DataFrame augmented with open-source sustainability scores and flags.
    """
    # -----------------------------------------------
    # 1) IMPORT OPEN SOURCE DATA & COMBINE 
    # -----------------------------------------------

    df_wba = generate_WBA()
    df_fe50 = generate_fe_50(vars_of_interest=vars_of_interest_FE_50)
    df_sbti = generate_SBTI(vars_of_interest=vars_of_interest_SBTI)
    df_spott = generate_SPOTT(vars_of_interest=vars_of_interest_SPOTT,
                              path_to_specific_questions=(PATH_TO_SPOTT_PALM_OIL_QUESTIONS,
                                                          PATH_TO_SPOTT_TIMBER_ETC_QUESTIONS,
                                                          PATH_TO_SPOTT_RUBBER_QUESTIONS),
                              questions_palm_oil=important_questions_palm_oil,
                              questions_timber=important_questions_timber,
                              questions_rubber=important_questions_rubber)
    # df_fairr = generate_FAIRR()
    df_tnfd = generate_TNFD()
    df_dat = generate_dat_scores(path_to_deforestation_action_tracker=PATH_TO_DEFORESTATION_ACTION_TRACKER)

    # Merge all the dataframes into one base dataframe
    '''
    - df_wba --> name, isin, sedol
    - df_sbti --> name, isin, LEI
    - df_fe50 --> name , (and GICS subindustry which may help with the mapping)
    - df_spott --> name, sedol, thomson reuters ticker
    - df_fairr --> name, isin  [TO BE IMPLEMENTED BY USER]
    - df_tnfd --> name
    - df_dat --> name
    '''

    # now, create a base of all companies that appear in at least one of the datasets
    # df_base = pd.concat([df_wba['company_name'], df_sbti['company_name'], df_fe50['company_name'],
    #                      df_spott['company_name'], df_fairr['company_name'], df_tnfd['company_name'],
    #                      df_dat['company_name']], ignore_index=True)
    df_base = pd.concat([df_wba['company_name'], df_sbti['company_name'], df_fe50['company_name'],
                         df_spott['company_name'], df_tnfd['company_name'],
                         df_dat['company_name']], ignore_index=True)

    # Delete duplicates and print number of deleted rows
    print('Number of duplicate company names collected from open-source datasets: ' + str(df_base.duplicated().sum()))
    df_base = df_base.drop_duplicates().to_frame()

    # Function to merge datasets based on different identifiers
    def merge_datasets(base_df, dataframes):
        """
        Merge multiple DataFrames into a base DataFrame.

        This function takes a base DataFrame and a list of tuples, where each tuple contains a DataFrame and a key
        column to merge on. It iteratively merges each DataFrame in the list with the base DataFrame using a left join.

        Args:
            base_df (pd.DataFrame): The base DataFrame onto which other DataFrames will be merged.
            dataframes (list of tuples): A list where each tuple contains a DataFrame and a key column name (str)
                                         to merge on. Example: [(df1, 'key1'), (df2, 'key2'), ...]

        Returns:
            pd.DataFrame: The merged DataFrame after all specified DataFrames have been merged onto the base DataFrame.
        """
        for df, on_key in dataframes:
            base_df = base_df.merge(df, on=on_key, how='left')
        return base_df

    # Define datasets_to_merge (incl. relevant identifiers)
    datasets_to_merge = [
        (df_wba[['company_name', 'isin', 'sedol']], 'company_name'),
        (df_sbti[['company_name', 'isin', 'lei']], 'company_name'),
        (df_spott[['company_name', 'sedol', 'thomson_reuters_ticker']], 'company_name'),
        # (df_fairr[['company_name', 'isin']], 'company_name'),
        (df_tnfd[['company_name']], 'company_name'),
        (df_dat[['company_name']], 'company_name')
    ]

    df_base = merge_datasets(df_base, datasets_to_merge)

    def combine_columns(df_before, prefs):
        """
        Combine multiple columns in a DataFrame based on specified prefixes.

        This function combines columns in a DataFrame that share the same prefix. For each prefix, it creates a new
        column that prioritizes non-NA values from the original columns with that prefix. The original columns are
        then dropped, and the new combined column is renamed to the original prefix.

        Args:
            df_before (pd.DataFrame): The DataFrame containing the columns to be combined.
            prefs (list of str): A list of prefixes to identify and combine columns.

        Returns:
            pd.DataFrame: The DataFrame with combined columns, where original columns with the specified prefixes
                          are replaced by a single column for each prefix, containing the first non-NA value in each row.
        """
        for pref in prefs:
            # Get columns with the specified prefix
            cols = [col for col in df_before.columns if col.startswith(pref)]

            # Combine columns, prioritizing non-NA values
            df_before[pref + '_combined'] = df_before[cols].bfill(axis=1).iloc[:, 0]

            # Drop the original columns
            df_before = df_before.drop(columns=cols)

            # Remove the prefix from the new column name
            df_before = df_before.rename(columns={pref + '_combined': pref})

        return df_before

    # Combine columns
    prefixes = ['isin', 'lei', 'sedol', 'thomson_reuters_ticker']
    df_base = combine_columns(df_base, prefixes)

    # Turn thomson_reuters_ticker= 'Private company' into NaN
    df_base['thomson_reuters_ticker'] = df_base['thomson_reuters_ticker'].replace('Private company', None)
    df_base.rename(columns={'thomson_reuters_ticker': 'ric'}, inplace=True)  # rename to ric (Reuters Instrument Code)

    # Adding variables to df_base # REMOVED FAIRR HERE!!
    datasets_to_add = [
        (df_wba, 'wba_', 'isin'),
        (df_sbti, 'sbti_', 'isin'),
        (df_fe50, 'fe50_', 'company_name'),
        (df_spott, 'spott_', 'company_name'),
        (df_tnfd, 'tnfd_', 'company_name'),
        (df_dat, 'dat_', 'company_name')
    ]

    for df, prefix, key in datasets_to_add:
        vars_of_interest = [key] + [col for col in df.columns if col.startswith(prefix)]
        df_base = df_base.merge(df[vars_of_interest], on=key, how='outer')

    # -----------------------------------------------
    # 2) COMBINE WITH PORTFOLIO DATA
    # -----------------------------------------------

    print('merging the open source scores to portfolio data')

    '''
    Now we need to add the collected variables to the master data.
    However, some companies are missing ISINs in the master data.
    
    Structure:
    2.0) Some prep work
    2.1) Merge based on ISINs
    2.2) Merge based on fuzzy string matching
    2.3) Combine the columns from the two merges
    '''

    # First, before splitting, add a cleaned company_name column as well as for the index weights
    df_base['company_name_clean'] = df_base['company_name'].apply(standardize_company_names)  # double check white space
    df_portfolio['company_name_clean'] = df_portfolio['name'].apply(standardize_company_names)

    identifiers_base = ['isin', 'lei', 'sedol', 'ric']
    identifiers_portfolio = ['permid', 'lei', 'ticker', 'ric']
    identifier_col = 'identifier'

    # Merge based on the entire set of company identifiers
    id_columns = set(identifiers_base).intersection(set(identifiers_portfolio))
    merged_df = df_base.copy()
    for id_col in id_columns:
        if id_col in df_base.columns and id_col in df_portfolio.columns:
            temp_df = df_base.merge(df_portfolio[[id_col, identifier_col]], on=id_col, how='left')
            merged_df = merged_df.combine_first(temp_df)
    matches_found = merged_df[identifier_col].notna().sum()
    print(f"Matches found after identifier merge: {matches_found} out of {len(df_base)}")

    # Find unmatched rows
    unmatched_base = merged_df[merged_df[identifier_col].isna()]
    unmatched_base.drop(columns=[identifier_col], inplace=True)  # drop identifier column from df_base
    unmatched_portfolio = df_portfolio[~df_portfolio[identifier_col].isin(merged_df[identifier_col].dropna())]

    # Merging based on Company Name
    merged_on_name = unmatched_base.merge(unmatched_portfolio[['company_name_clean', identifier_col]],
                                          on='company_name_clean', how='left')
    merged_df.update(merged_on_name)
    matches_found = merged_df[identifier_col].notna().sum()
    print(f"Matches found after company name merge: {matches_found} out of {len(df_base)}")

    # Find unmatched rows after company name merge
    unmatched_base = merged_df[merged_df[identifier_col].isna()]
    unmatched_base.drop(columns=[identifier_col], inplace=True)  # drop identifier column from df_base
    unmatched_portfolio = df_portfolio[~df_portfolio[identifier_col].isin(merged_df[identifier_col].dropna())]

    # FUZZY STRING MATCHING VIA RAPIDFUZZ
    list_unmatched_base = unmatched_base['company_name_clean'].tolist()
    list_unmatched_portfolio = unmatched_portfolio['company_name_clean'].tolist()

    from utils import find_fuzzy
    from rapidfuzz import fuzz
    fuzzy_matches = find_fuzzy(list_unmatched_base, list_unmatched_portfolio, score_cutoff=string_matching_threshold,
                               scorer=fuzz.ratio)

    # Apply fuzzy matches
    for match in fuzzy_matches:
        base_index = unmatched_base.index[match['df_master_index']]
        portfolio_index = unmatched_portfolio.index[match['asset_data_index']]
        identifier_value = unmatched_portfolio.at[portfolio_index, identifier_col]
        merged_df.at[base_index, identifier_col] = identifier_value

    matches_found = merged_df[identifier_col].notna().sum()
    print(f"Matches found after fuzzy merge: {matches_found} out of {len(df_base)}")
    print('done merging the scores')

    # Add positive flags for each data_source
    tnfd_flag = ['tnfd_early_adopter']
    wba_flag = ['wba_NAT.B01', 'wba_NAT.B02', 'wba_NAT.B06', 'wba_NAT.C02', 'wba_NAT.C05', 'wba_NAT.C07',
                'wba_NAT.B03.ED', 'wba_NAT.B05.EA', 'wba_NAT.B05.EB', 'wba_NAT.B05.EC', 'wba_NAT.B05.EG',
                'wba_NAT.B05.EJ', 'wba_NAT.C05.EA', 'wba_ma2:_ecosystems_and_biodiversity',
                'wba_ma3:_social_inclusion_and_community_impact']
    sbti_flag = ['sbti_long_term_target', 'sbti_near_term_target']

    spott_flag = ['spott_sust_policy_score', 'spott_landbank_score', 'spott_cert_standards_score',
                  'spott_def_biodiv_score', 'spott_hcv_hcs_score', 'spott_soils_fire_score',
                  'spott_community_land_labour_score', 'spott_smallholders_suppliers_score',
                  'spott_gov_grievance_score', 'spott_rspo_member', 'spott_palm_oil_1', 'spott_palm_oil_2',
                  'spott_palm_oil_55', 'spott_palm_oil_57', 'spott_palm_oil_129', 'spott_palm_oil_130',
                  'spott_timber_1', 'spott_timber_2', 'spott_timber_52', 'spott_timber_54', 'spott_timber_130',
                  'spott_timber_131', 'spott_rubber_1', 'spott_rubber_2', 'spott_rubber_55', 'spott_rubber_57',
                  'spott_rubber_130', 'spott_rubber_131']
    # fairr_flag = ['fairr_df_target_soy', 'fairr_df_target_cattle', 'fairr_engagement_soy',
    #               'fairr_engagement_cattle', 'fairr_feed_ingredients_ratios', 'fairr_feed_innovation',
    #               'fairr_ecosystem_impacts', 'fairr_deforestation_score', 'fairr_work_score']

    # Create positive flags for each open source dataset
    merged_df['tnfd_positive_flag'] = merged_df.apply(lambda row: 1 if any(row[col] > 0 for col in tnfd_flag) else 0,
                                                      axis=1)
    merged_df['wba_positive_flag'] = merged_df.apply(lambda row: 1 if any(row[col] > 0 for col in wba_flag) else 0,
                                                    axis=1)
    merged_df['sbti_positive_flag'] = merged_df.apply(
        lambda row: 1 if any(pd.notna(row[col]) for col in sbti_flag) else 0, axis=1)
    merged_df['spott_positive_flag'] = merged_df.apply(
        lambda row: 1 if any(pd.notna(row[col]) for col in spott_flag) and row[spott_flag[9]] != 'No' else 0, axis=1)
    # merged_df['fairr_positive_flag'] = merged_df.apply(lambda row: 1 if any(pd.notna(row[col]) for col in fairr_flag) and any(row[col] > 0 for col in fairr_flag) else 0, axis=1)

    open_source_flag_columns = [col for col in df_base.columns if col.endswith('positive_flag')]

    # Create a column that indicated whether an ISIN has any positive flag
    merged_df['positive_flag'] = merged_df.apply(
        lambda row: 1 if any(row[col] > 0 for col in open_source_flag_columns) else 0, axis=1)

    # Reduce dataset to "identifier" and with prefixes of the open source datasets
    # open_source_prefixes = ['tnfd_', 'wba_', 'sbti_', 'spott_', 'fairr_', 'dat_']
    open_source_prefixes = ['tnfd_', 'wba_', 'sbti_', 'spott_', 'dat_']
    prefix_columns = [col for col in merged_df.columns if
                      any(col.startswith(prefix) for prefix in open_source_prefixes)]
    merged_df_final = merged_df[['identifier'] + prefix_columns + open_source_flag_columns + ['positive_flag']]

    # Only keep those with matched identifiers
    merged_df_final = merged_df_final[merged_df_final['identifier'].notna()]

    # Combine rows with the same identifier
    merged_df_final = merged_df_final.groupby('identifier').agg(
        lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan).reset_index()

    # Convert all columns except 'identifier' to integer
    # Columns to exclude
    exclude_columns = ['identifier', 'sbti_long_term_target', 'sbti_near_term_target',
                       'fe50_gics_sub-industry',
                       'fe50_metric_1b:_scope_3_from_agriculture',
                       'fe50_metric_1c:_scope_3_from_land_use_change',
                       'fe50_metric_5d:_time-bound_commitment_to_achieve_a_deforestation_and_conversion_free_supply_chain_by_2025_across_the_business',
                       'spott_parent_company',
                       'spott_rspo_member']

    # Transform columns to numeric
    for col in merged_df_final.columns:
        if col not in exclude_columns:
            merged_df_final[col] = pd.to_numeric(merged_df_final[col], errors='coerce')

    return merged_df_final
