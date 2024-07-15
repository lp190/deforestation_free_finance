"""
Description: 
    This script implements the decision tree 2 logic to classify companies into high, medium, low policy risk.
    It processes data from various sources, applies the DT2 logic to determine policy risk, and exports the results.

    !!! IMPORTANT NOTE: This file only runs if the user already ran the run_decision_tree1.py script !!!

Outline of the script:
    1) Load and merge data from multiple sources including open source, CDP, Forest500, and LSEG (Refinitiv).
    2) Create DT2 risk buckets based on decision tree logic and assign companies to high, medium, or low policy risk.
    3) Export the final evaluation and relevant data to a CSV file.

"""

# import packages
import os
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

from apply.apply_open_source_scores import apply_open_source_scores
from filepaths import PATH_TO_OUTPUT_FOLDER, PATH_TO_DT1
from generate.generate_cdp import generate_cdp
from prep.prep_forest500 import prep_forest500
from prep.prep_proprietary_policy_vars import prep_proprietary_policy_vars


def apply_dt2_policy_risk(df_portfolio,
                          cdp_data_exists,
                          proprietary_policy_data_exists,
                          perform_full_analysis,
                          fuzzy_match_cutoff_F500,
                          forest500_threshold_bucketing_dt2,
                          refinitiv_hr_score_threshold_strong,
                          spott_community_land_labour_score_threshold_strong,
                          spott_smallholders_suppliers_score_threshold_strong,
                          spott_gov_grievance_score_threshold_strong,
                          wba_ma3_social_inclusion_and_community_impact_threshold_strong,
                          fairr_work_score_threshold_strong,
                          refinitiv_hr_score_threshold_mediocre_or_weak,
                          spott_community_land_labour_score_threshold_mediocre_or_weak,
                          spott_smallholders_suppliers_score_threshold_mediocre_or_weak,
                          spott_gov_grievance_score_threshold_mediocre_or_weak,
                          wba_ma3_social_inclusion_and_community_impact_threshold_mediocre_or_weak,
                          fairr_work_score_threshold_mediocre_or_weak,
                          spott_def_biodiv_score_threshold_strong,
                          wba_ma2_ecosystems_and_biodiversity_threshold_strong,
                          dat_score_threshold_strong,
                          fairr_deforestation_score_threshold_strong,
                          spott_def_biodiv_score_threshold_mediocre_or_weak,
                          wba_ma2_ecosystems_and_biodiversity_threshold_mediocre_or_weak,
                          dat_score_threshold_mediocre_or_weak,
                          fairr_deforestation_score_threshold_mediocre_or_weak
                          ):
    """
    Apply policy risk assessment (DT2) to the input DataFrame and classify companies into risk buckets (high, medium, low).

    Args:
        df_portfolio (pd.DataFrame): DataFrame containing at least the column 'isin'.
        cdp_data_exists (bool): Whether the user has access to CDP data.
        proprietary_policy_data_exists (bool): Whether the user has access to proprietary policy data.
        perform_full_analysis (bool): Whether to perform a full analysis or use existing data.
        fuzzy_match_cutoff_F500 (float): Fuzzy match cutoff for Forest500.
        forest500_threshold_bucketing_dt2 (float): Threshold for Forest500 total score.
        refinitiv_hr_score_threshold_strong (float): Strong threshold for Refinitiv human rights score.
        spott_community_land_labour_score_threshold_strong (float): Strong threshold for SPOTT community, land, and labour score.
        spott_smallholders_suppliers_score_threshold_strong (float): Strong threshold for SPOTT smallholders and suppliers score.
        spott_gov_grievance_score_threshold_strong (float): Strong threshold for SPOTT governance and grievance score.
        wba_ma3_social_inclusion_and_community_impact_threshold_strong (float): Strong threshold for WBA social inclusion and community impact score.
        fairr_work_score_threshold_strong (float): Strong threshold for FAIRR work score.
        refinitiv_hr_score_threshold_mediocre_or_weak (float): Mediocre or weak threshold for Refinitiv human rights score.
        spott_community_land_labour_score_threshold_mediocre_or_weak (float): Mediocre or weak threshold for SPOTT community, land, and labour score.
        spott_smallholders_suppliers_score_threshold_mediocre_or_weak (float): Mediocre or weak threshold for SPOTT smallholders and suppliers score.
        spott_gov_grievance_score_threshold_mediocre_or_weak (float): Mediocre or weak threshold for SPOTT governance and grievance score.
        wba_ma3_social_inclusion_and_community_impact_threshold_mediocre_or_weak (float): Mediocre or weak threshold for WBA social inclusion and community impact score.
        fairr_work_score_threshold_mediocre_or_weak (float): Mediocre or weak threshold for FAIRR work score.
        spott_def_biodiv_score_threshold_strong (float): Strong threshold for SPOTT deforestation and biodiversity score.
        wba_ma2_ecosystems_and_biodiversity_threshold_strong (float): Strong threshold for WBA ecosystems and biodiversity score.
        dat_score_threshold_strong (float): Strong threshold for DAT score.
        fairr_deforestation_score_threshold_strong (float): Strong threshold for FAIRR deforestation score.
        spott_def_biodiv_score_threshold_mediocre_or_weak (float): Mediocre or weak threshold for SPOTT deforestation and biodiversity score.
        wba_ma2_ecosystems_and_biodiversity_threshold_mediocre_or_weak (float): Mediocre or weak threshold for WBA ecosystems and biodiversity score.
        dat_score_threshold_mediocre_or_weak (float): Mediocre or weak threshold for DAT score.
        fairr_deforestation_score_threshold_mediocre_or_weak (float): Mediocre or weak threshold for FAIRR deforestation score.

    Returns:
        pd.DataFrame: The input DataFrame with added policy risk assessments and classifications.
    """

    # Load the DT2 results from output folder (internal_data).
    # If it's not there, then generate it.
    policy_risk_path = Path(PATH_TO_OUTPUT_FOLDER) / 'internal_data' / 'policy_risk.csv'

    if policy_risk_path.exists() and not perform_full_analysis:
        print('loading: policy_risk.csv')
        df_policy_risk = pd.read_csv(policy_risk_path)
    else:
        # LOAD OPEN SOURCE DATA
        print("Compiling open source data...")
        df_open_source = apply_open_source_scores(df_portfolio)

        # LOAD FOREST500 DATA  
        print("Compiling Forest500 data...")
        path_to_f500 = Path(PATH_TO_OUTPUT_FOLDER) / 'internal_data' / 'forest500_matches.csv'
        if path_to_f500.exists() and not perform_full_analysis:
            forest500 = pd.read_csv(path_to_f500)
            forest500['identifier'] = forest500['permid'].copy()  # create identifier column
        else:
            columns_to_load_forest500_comps = ('Company', 'HQ', 'Total Score /100')
            columns_to_load_forest500_fis = ('FI name', 'FI Headquarters', 'Total Score / 100')
            manual_false_positive_list = (
                'bank of changsha', 'smc', 'shizuoka financial', 'misumi', 'rogersmmunications b',
                'banco bradesco', 'china baoan', 'bank of chengdu', 'obic',
                'sainsbury j', 'american financial', 'china citic bank')
            forest500 = prep_forest500(df_portfolio, columns_comps=columns_to_load_forest500_comps,
                                       columns_fis=columns_to_load_forest500_fis,
                                       fuzzy_match_cutoff=fuzzy_match_cutoff_F500,
                                       false_positive_list=manual_false_positive_list,
                                       manual_matches=True)
            forest500['identifier'] = forest500['permid'].copy()  # create identifier column

        # LOAD CDP DATA
        if cdp_data_exists:
            print("Compiling CDP data...")
            df_cdp_policy_risk = generate_cdp()  # creates the cdp_pol.csv file (by default a placeholder)

        # LOAD PROPRIETARY DATA on human rights or deforestation policy variables
        if proprietary_policy_data_exists:
            print("Compiling proprietary policy data...")
            df_proprietary_policy_vars = prep_proprietary_policy_vars()

        # LOAD DT1 OUTPUT
        # Check if the 'output_final_dt1.csv' file exists

        if PATH_TO_DT1.exists():
            dt1_output = pd.read_excel(PATH_TO_DT1,
                                       usecols=['company_name', 'identifier', 'country_name', 'country_iso',
                                                'nace_code', 'nace_desc', 'dt1_bucket']
                                       ).astype({'identifier': 'str'})
        else:
            raise ValueError('DT1 not found, please run the run_decision_tree1.py script first')

        # MERGE THE THREE DATASETS (using the reduce function from functools)
        print("Merge all datasets with output from DT1...")
        dfs_to_merge = [forest500, df_open_source]

        # Conditionally add df_proprietary_policy_vars and cdp_policy_risk if they exist
        if 'df_proprietary_policy_vars' in locals():
            dfs_to_merge.append(df_proprietary_policy_vars)

        if 'df_cdp_policy_risk' in locals():
            dfs_to_merge.append(df_cdp_policy_risk)

        df_policy_risk_merged = reduce(lambda left, right: pd.merge(left, right, on='identifier', how='outer'),
                                       dfs_to_merge)

        # Merge to DT1 output
        df_policy_risk = pd.merge(dt1_output, df_policy_risk_merged, on='identifier', how='left')

        # -----------------------------------------------
        # 2) CREATE DT2 BUCKETS
        # -----------------------------------------------
        print("Creating DT2 buckets...")

        # Create DT2 bucket
        df_policy_risk['dt2_bucket'] = np.nan
        df_policy_risk['dt2_bucket'] = df_policy_risk['dt2_bucket'].astype('object')

        # Assign respective values according to Global Canopy's et al decision tree 2

        # Step 0 - Based on DT1 bucket
        # If the deforestation exposure is modelled as low, then the policy risk is assumed to be low as well
        df_policy_risk.loc[df_policy_risk['dt1_bucket'] == 'low', 'dt2_bucket'] = 'low'
        # Drop dt1_bucket
        df_policy_risk = df_policy_risk.drop('dt1_bucket', axis=1)

        # Step 1 - Based on total score by Forest500
        df_policy_risk.loc[
            df_policy_risk['forest500_total_score'] >= forest500_threshold_bucketing_dt2, 'dt2_bucket'] = 'medium'
        df_policy_risk.loc[
            df_policy_risk['forest500_total_score'] < forest500_threshold_bucketing_dt2, 'dt2_bucket'] = 'high'

        # Step 2 - Based on human rights and deforestation policies
        print("Evaluate existence of deforestation and human rights policies...")
        # Create new variable which equals 1 if company has a deforestation policy, 0 if not 
        # and NA if the company does not appear in any of our datasets
        df_policy_risk['deforestation_policy_in_place'] = np.nan
        df_policy_risk['human_rights_policy_in_place'] = np.nan

        # Define set of policy variables that point at the existence of a deforestation or human rights policy

        cdp_deforestation_policy_variables = []  # ADD CDP VARIABLES (ensure they are binary 0/1 variables)
        proprietary_deforestation_policy_variables = []  # ADD PROPRIETARY VARIABLES (ensure they are binary 0/1 variables)
        proprietary_hr_policy_variables = []  # ADD PROPRIETARY VARIABLES (ensure they are binary 0/1 variables)

        deforestation_policy_variables = ['wba_NAT.B01', 'wba_NAT.B02', 'wba_NAT.B06', 'wba_NAT.C02',
                                          'wba_NAT.C05', 'wba_NAT.C07', 'wba_NAT.B03.ED', 'wba_NAT.B05.EA',
                                          'wba_NAT.B05.EB', 'wba_NAT.B05.EC', 'wba_NAT.B05.EG', 'wba_NAT.B05.EJ',
                                          'spott_palm_oil_1', 'spott_palm_oil_2', 'spott_palm_oil_55',
                                          'spott_palm_oil_57', 'spott_timber_1', 'spott_timber_2', 'spott_timber_52',
                                          'spott_timber_54', 'spott_rubber_1', 'spott_rubber_2', 'spott_rubber_55',
                                          'spott_rubber_57', 'dat__1.1_score'] + \
                                         cdp_deforestation_policy_variables + \
                                         proprietary_deforestation_policy_variables

        hr_policy_variables = ['wba_NAT.C05.EA', 'spott_palm_oil_129', 'spott_palm_oil_130', 'spott_timber_130',
                               'spott_timber_131', 'spott_rubber_130', 'spott_rubber_131', 'dat_palm_oil_4.3_score',
                               'dat_soy_4.3_score', 'dat_beef_leather_4.3_score', 'dat_timber,_pulp_paper_4.3_score'] + \
                              proprietary_hr_policy_variables

        # Function to assign 0 or 1 to 'deforestation_policy_in_place'
        def check_if_any_policy_policy_exists(row, policy_variables):
            """
           Check if any policy exists for the given row based on specified policy variables.

           Args:
               row (pd.Series): A row of data from a DataFrame.
               policy_variables (list): List of column names representing policy variables to check.

           Returns:
               int or float: Returns 1 if any policy exists, 0 if no policies exist, and NaN if all policy variables are NaN.
           """
            if all(pd.isna(row[policy_variables])):
                return np.nan
            elif all(pd.isna(row[policy_variables]) | (row[policy_variables] == 0)):
                return 0
            else:
                return 1

        # Apply the function to each row
        df_policy_risk['deforestation_policy_in_place'] = df_policy_risk.apply(
            lambda row: check_if_any_policy_policy_exists(row, deforestation_policy_variables),
            axis=1
        )

        df_policy_risk['human_rights_policy_in_place'] = df_policy_risk.apply(
            lambda row: check_if_any_policy_policy_exists(row, hr_policy_variables),
            axis=1
        )

        # the conditions are structured by first checking that the company was not assigned in Step1, if so then
        # second checking if the company has both a good hr and deforestation policy in place. If the company has no, or
        # either only a hr/deforestation policy then it will be assigned to the high risk bucket
        df_policy_risk.loc[df_policy_risk['dt2_bucket'].isna() &
                           ((df_policy_risk['human_rights_policy_in_place'] == 0) |
                            (df_policy_risk['deforestation_policy_in_place'] == 0)), 'dt2_bucket'] = 'high'

        # Step 3 & 4 - Based on strength of human rights and deforestation policies
        print("Evaluate strength of deforestation and human rights policies...")
        # we classify all companies with both a deforestation and hr policy into high, medium and low policy risk
        # buckets based on the strength of their policies.
        # if there is only a very bad policy in place: =0
        # if there is normal/good policy in place: =1
        # if there is a very good policy in place: =2

        df_policy_risk['strength_deforestation_policy'] = np.nan
        df_policy_risk['strength_hr_policy'] = np.nan

        #######
        def check_condition(df, col_name, threshold, op='>='):
            """
            Helper function to check if condition exists, to prevent key errors

            This function evaluates a condition on a given column of a DataFrame and returns a boolean Series indicating
            whether each row meets the condition. The condition is specified by a threshold and an operator ('>=' or '<').

            Args:
                df (pd.DataFrame): The DataFrame to be checked.
                col_name (str): The name of the column to evaluate.
                threshold (numeric): The threshold value for the condition.
                op (str): The operator for the condition, either '>=' (default) or '<'.

            Returns:
                pd.Series: A boolean Series where True indicates that the condition is met for the corresponding row,
                           and False otherwise. If the column does not exist, a Series of False values is returned.
            """
            if col_name in df.columns:
                if op == '>=':
                    return df[col_name] >= threshold
                elif op == '<':
                    return df[col_name] < threshold
            else:
                return pd.Series([False] * len(df))

        ########

        # Define the decision rules with checks for column existence
        conditions_hr_strength = [
            (df_policy_risk['dt2_bucket'].isna() &
             (check_condition(df_policy_risk, 'refinitiv_hr_score', refinitiv_hr_score_threshold_strong) |
              check_condition(df_policy_risk, 'spott_community_land_labour_score',
                              spott_community_land_labour_score_threshold_strong) |
              check_condition(df_policy_risk, 'spott_smallholders_suppliers_score',
                              spott_smallholders_suppliers_score_threshold_strong) |
              check_condition(df_policy_risk, 'spott_gov_grievance_score', spott_gov_grievance_score_threshold_strong) |
              check_condition(df_policy_risk, 'wba_ma3:_social_inclusion_and_community_impact',
                              wba_ma3_social_inclusion_and_community_impact_threshold_strong) |
              check_condition(df_policy_risk, 'fairr_work_score', fairr_work_score_threshold_strong))),

            (df_policy_risk['dt2_bucket'].isna() &
             (check_condition(df_policy_risk, 'refinitiv_hr_score', refinitiv_hr_score_threshold_mediocre_or_weak) |
              check_condition(df_policy_risk, 'spott_community_land_labour_score',
                              spott_community_land_labour_score_threshold_mediocre_or_weak) |
              check_condition(df_policy_risk, 'spott_smallholders_suppliers_score',
                              spott_smallholders_suppliers_score_threshold_mediocre_or_weak) |
              check_condition(df_policy_risk, 'spott_gov_grievance_score',
                              spott_gov_grievance_score_threshold_mediocre_or_weak) |
              check_condition(df_policy_risk, 'wba_ma3:_social_inclusion_and_community_impact',
                              wba_ma3_social_inclusion_and_community_impact_threshold_mediocre_or_weak) |
              check_condition(df_policy_risk, 'fairr_work_score', fairr_work_score_threshold_mediocre_or_weak))),

            (df_policy_risk['dt2_bucket'].isna() &
             (check_condition(df_policy_risk, 'refinitiv_hr_score', refinitiv_hr_score_threshold_mediocre_or_weak,
                              '<') |
              check_condition(df_policy_risk, 'spott_community_land_labour_score',
                              spott_community_land_labour_score_threshold_mediocre_or_weak, '<') |
              check_condition(df_policy_risk, 'spott_smallholders_suppliers_score',
                              spott_smallholders_suppliers_score_threshold_mediocre_or_weak, '<') |
              check_condition(df_policy_risk, 'spott_gov_grievance_score',
                              spott_gov_grievance_score_threshold_mediocre_or_weak, '<') |
              check_condition(df_policy_risk, 'wba_ma3:_social_inclusion_and_community_impact',
                              wba_ma3_social_inclusion_and_community_impact_threshold_mediocre_or_weak, '<') |
              check_condition(df_policy_risk, 'fairr_work_score', fairr_work_score_threshold_mediocre_or_weak, '<'))),
        ]

        # Corresponding values to assign for each condition
        values = [2, 1, 0]

        # Apply the conditions and assign values to 'strength_hr_policy'
        df_policy_risk['strength_hr_policy'] = np.select(conditions_hr_strength, values, default=np.nan)

        #####

        conditions_defo_strength = [
            (df_policy_risk['dt2_bucket'].isna() &
             (check_condition(df_policy_risk, 'spott_def_biodiv_score', spott_def_biodiv_score_threshold_strong) |
              check_condition(df_policy_risk, 'wba_ma2:_ecosystems_and_biodiversity',
                              wba_ma2_ecosystems_and_biodiversity_threshold_strong) |
              check_condition(df_policy_risk, 'dat_score', dat_score_threshold_strong) |
              check_condition(df_policy_risk, 'fairr_deforestation_score',
                              fairr_deforestation_score_threshold_strong))),

            (df_policy_risk['dt2_bucket'].isna() &
             (check_condition(df_policy_risk, 'spott_def_biodiv_score',
                              spott_def_biodiv_score_threshold_mediocre_or_weak) |
              check_condition(df_policy_risk, 'wba_ma2:_ecosystems_and_biodiversity',
                              wba_ma2_ecosystems_and_biodiversity_threshold_mediocre_or_weak) |
              check_condition(df_policy_risk, 'dat_score', dat_score_threshold_mediocre_or_weak) |
              check_condition(df_policy_risk, 'fairr_deforestation_score',
                              fairr_deforestation_score_threshold_mediocre_or_weak))),

            (df_policy_risk['dt2_bucket'].isna() &
             (check_condition(df_policy_risk, 'spott_def_biodiv_score',
                              spott_def_biodiv_score_threshold_mediocre_or_weak, '<') |
              check_condition(df_policy_risk, 'wba_ma2:_ecosystems_and_biodiversity',
                              wba_ma2_ecosystems_and_biodiversity_threshold_mediocre_or_weak, '<') |
              check_condition(df_policy_risk, 'dat_score', dat_score_threshold_mediocre_or_weak, '<') |
              check_condition(df_policy_risk, 'fairr_deforestation_score',
                              fairr_deforestation_score_threshold_mediocre_or_weak, '<'))),
        ]

        # Corresponding values to assign for each condition
        values = [2, 1, 0]

        # Apply the conditions and assign values to 'strength_deforestation_policy'
        df_policy_risk['strength_deforestation_policy'] = np.select(conditions_defo_strength, values, default=np.nan)

        # low risk bucket if company has at least good hr policy and very good deforestation policy or
        # at least a good deforestation policy and very good hr policy
        df_policy_risk.loc[df_policy_risk['dt2_bucket'].isna() &
                           ((df_policy_risk['human_rights_policy_in_place'] == 1) &
                            (df_policy_risk['deforestation_policy_in_place'] == 1)) &
                           (((df_policy_risk['strength_deforestation_policy'] == 2) &
                             (df_policy_risk['strength_hr_policy'] >= 1)) |
                            ((df_policy_risk['strength_deforestation_policy'] >= 1) &
                             (df_policy_risk['strength_hr_policy'] == 2))), 'dt2_bucket'] = 'low'

        # medium risk bucket if company has at least good hr policy and at least good deforestation policy
        df_policy_risk.loc[df_policy_risk['dt2_bucket'].isna() &
                           ((df_policy_risk['human_rights_policy_in_place'] == 1) &
                            (df_policy_risk['deforestation_policy_in_place'] == 1)) &
                           ((df_policy_risk['strength_deforestation_policy'] >= 1) &
                            (df_policy_risk['strength_hr_policy'] >= 1)), 'dt2_bucket'] = 'medium'

        # high risk bucket if company has neither a good hr policy nor a good deforestation policy
        df_policy_risk.loc[df_policy_risk['dt2_bucket'].isna() &
                           ((df_policy_risk['human_rights_policy_in_place'] == 1) &
                            (df_policy_risk['deforestation_policy_in_place'] == 1)) &
                           ((df_policy_risk['strength_deforestation_policy'] == 0) |
                            (df_policy_risk['strength_hr_policy'] == 0)), 'dt2_bucket'] = 'high'

        # We set all companies which have no risk bucket assigned due to missing data into the medium or high bucket
        # df_policy_risk.loc[(df_policy_risk['dt2_bucket'].isna()), 'dt2_bucket'] = 'medium'
        # df_policy_risk.loc[(df_policy_risk['dt2_bucket'].isna()), 'dt2_bucket'] = 'high'

        # OR we set all companies which have no risk bucket assigned due to missing data into the following buckets:
        df_policy_risk.loc[df_policy_risk['dt2_bucket'].isna() &
                           ((df_policy_risk['human_rights_policy_in_place'] == 1) &
                            (df_policy_risk['deforestation_policy_in_place'] == 1)), 'dt2_bucket'] = 'medium'
        df_policy_risk.loc[(df_policy_risk['dt2_bucket'].isna()), 'dt2_bucket'] = 'high'

        # -----------------------------------------------
        # 3) EXPORT & COSMETICS
        # -----------------------------------------------

        # rename columns via dictionary
        new_column_names = {
            'forest500_total_score': 'forest500_score'
        }

        # rename columns
        df_policy_risk = df_policy_risk.rename(columns=new_column_names)

        # Export to csv
        df_policy_risk.to_csv(os.path.join(PATH_TO_OUTPUT_FOLDER, 'internal_data/dt2_policy_risk.csv'), index=False)

    return df_policy_risk


### SCORING FUNCTION


def apply_dt2_weighted_average_approach(portfolio_results_dt2,
                                        weight_dictionary,
                                        cutoffs=(0.9, 0.7)):
    """
    Apply a weighted average approach to assess policy risk (DT2) levels of companies in a portfolio.

    Args:
        portfolio_results_dt2 (DataFrame): DataFrame containing the DT2 portfolio data.
        weight_dictionary (dict): Dictionary of weights for DT2-related scores.
        cutoffs (tuple): Quantile cutoffs for determining risk levels (default is (0.9, 0.7)).

    Returns:
        tuple: Three DataFrames containing the low, medium, and high-risk data, respectively.
    """

    def log_transform_and_normalize(series):
        """
        Log-transform and normalize a series, handling zeros appropriately and inverting the result.

        Args:
            series (Series): Series to be transformed and normalized.

        Returns:
            Series: Transformed, normalized, and inverted series.
        """
        # Replace NaNs with zeros
        series = series.fillna(0)

        non_zero = series > 0
        log_transformed = np.log1p(series)  # Use np.log1p to handle zeros properly

        if non_zero.sum() == 0:
            # If all values are zero, return the original series
            return series

        # Normalize the log_transformed non-zero values
        min_val = log_transformed[non_zero].min()
        max_val = log_transformed[non_zero].max()

        if min_val == max_val:
            # If all non-zero values are the same, return series of zeros and ones
            normalized = pd.Series(0, index=series.index)
            normalized[non_zero] = 1
        else:
            normalized = pd.Series(0, index=series.index)
            normalized[non_zero] = (log_transformed[non_zero] - min_val) / (max_val - min_val)
            # change any zero values in normalized[non_zero] to half of the min value in normalized[non_zero]
            # Sort the series
            sorted_series = normalized[non_zero].sort_values()
            # Get the unique values to avoid duplicates
            unique_values = sorted_series.unique()
            # Retrieve the second-smallest value
            second_smallest_value = unique_values[1]
            # Now assign the lowest value in the non-zero series the second smallest normalized value divided by 2
            normalized[non_zero] = normalized[non_zero].replace(0, second_smallest_value / 2)

        # Invert the normalized values
        inverted = 1 - normalized

        return inverted

    # Because some columns are not yet between 0 and 1 we apply a log transformation followed by a normalization
    columns_to_normalize = [
        'human_rights_policy_in_place', 'deforestation_policy_in_place', 'forest500_score',
        'strength_hr_policy', 'strength_deforestation_policy'
    ]
    # Apply transformation and normalization, replacing zeros
    portfolio_results_dt2_scores = portfolio_results_dt2.copy()
    for column in columns_to_normalize:
        portfolio_results_dt2_scores[column] = log_transform_and_normalize(portfolio_results_dt2_scores[column])

    def calculate_weighted_score(df, weight_dict):
        """
        Calculate the weighted score for a DataFrame based on a given weight dictionary.

        Args:
            df (DataFrame): DataFrame containing the data.
            weight_dict (dict): Dictionary of weights.

        Returns:
            float: Weighted score.
        """
        for key in weight_dict:
            if key not in df:
                df[key] = 0
        weighted_score = sum(df[key] * weight for key, weight in weight_dict.items())
        return weighted_score

    # Apply the weighting
    portfolio_results_dt2_scores['dt2_score'] = portfolio_results_dt2_scores.apply(
        lambda row: calculate_weighted_score(row, weight_dictionary), axis=1)

    # Normalize the dt2_score
    portfolio_results_dt2_scores['dt2_score'] = log_transform_and_normalize(portfolio_results_dt2_scores['dt2_score'])

    # Rank based on the score
    portfolio_results_dt2_scores = portfolio_results_dt2_scores.sort_values(
        by='dt2_score', ascending=False)

    # Subset the df only including identifier and dt2_score
    portfolio_results_dt2_scores = portfolio_results_dt2_scores[['identifier', 'dt2_score']]

    # Now do the quantiles
    cutoff_high = portfolio_results_dt2_scores['dt2_score'].quantile(cutoffs[0])
    cutoff_medium = portfolio_results_dt2_scores['dt2_score'].quantile(cutoffs[1])
    if cutoff_medium == 0:
        cutoff_medium = np.mean(portfolio_results_dt2_scores['dt2_score'])

    portfolio_results_dt2_scores['dt2_bucket_based_on_score'] = 'low'
    portfolio_results_dt2_scores.loc[
        portfolio_results_dt2_scores['dt2_score'] >= cutoff_medium, 'dt2_bucket_based_on_score'] = 'medium'
    portfolio_results_dt2_scores.loc[
        portfolio_results_dt2_scores['dt2_score'] >= cutoff_high, 'dt2_bucket_based_on_score'] = 'high'

    return portfolio_results_dt2_scores[['identifier', 'dt2_score', 'dt2_bucket_based_on_score']]
