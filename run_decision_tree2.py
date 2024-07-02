"""
Description:
    This code compiles all the datapoints linked to DT2.
    Including the suggestions for the buckets.

OUTPUT
    Excel file: output_final_dt2.xlsx
"""
# Import packages
import os

import pandas as pd

from apply.apply_decision_tree2 import apply_dt2_policy_risk, apply_dt2_weighted_average_approach
from filepaths import PATH_TO_OUTPUT_FOLDER, PATH_TO_PORTFOLIO_DATA
from user_input import (
    # GENERAL SETTINGS
    perform_full_analysis,
    FUZZY_MATCH_CUTOFF_F500,
    cdp_data_exists,
    proprietary_policy_data_exists,
    # DT2, Step 1: Threshold F500 score
    forest500_threshold_bucketing_dt2,  # DT2, Step 1: Threshold F500 score
    # DT2, Step 2: No user input needed.

    # DT2, Step3 : Human rights and deforestation policy strength
    # Define thresholds for companies with strong human rights policies
    refinitiv_hr_score_threshold_strong,
    spott_community_land_labour_score_threshold_strong,
    spott_smallholders_suppliers_score_threshold_strong,
    spott_gov_grievance_score_threshold_strong,
    wba_ma3_social_inclusion_and_community_impact_threshold_strong,
    fairr_work_score_threshold_strong,
    # Define thresholds for companies with mediocre (if >=) or weak (if <) human rights policies
    refinitiv_hr_score_threshold_mediocre_or_weak,
    spott_community_land_labour_score_threshold_mediocre_or_weak,
    spott_smallholders_suppliers_score_threshold_mediocre_or_weak,  # check
    spott_gov_grievance_score_threshold_mediocre_or_weak,
    wba_ma3_social_inclusion_and_community_impact_threshold_mediocre_or_weak,
    fairr_work_score_threshold_mediocre_or_weak,
    # Define thresholds for companies with strong deforestation policies
    spott_def_biodiv_score_threshold_strong,
    wba_ma2_ecosystems_and_biodiversity_threshold_strong,
    dat_score_threshold_strong,
    fairr_deforestation_score_threshold_strong,
    # Define thresholds for companies with mediocre (if >=) or weak (if <) deforestation policies
    spott_def_biodiv_score_threshold_mediocre_or_weak,
    wba_ma2_ecosystems_and_biodiversity_threshold_mediocre_or_weak,
    dat_score_threshold_mediocre_or_weak,
    fairr_deforestation_score_threshold_mediocre_or_weak,
)
from utils import clean_df_portfolio

# Load df_portfolio (by default: MSCI ACWI Index with limited data)
df_portfolio = pd.read_excel(PATH_TO_PORTFOLIO_DATA, sheet_name='data').astype(str)
df_portfolio = clean_df_portfolio(df_portfolio)  # apply clean_df_portfolio from utils
df_portfolio["identifier"] = df_portfolio["permid"].astype(str)  # define identifier column

### -------------------------------------------------------- ###
###         STEP 1: Attach DT2 variables                     ###
### -------------------------------------------------------- ###

"Collect DT2 Variables & Assign Buckets"
# apply function from apply_decision_tree2.py
df_portfolio = apply_dt2_policy_risk(df_portfolio, cdp_data_exists, proprietary_policy_data_exists,
                                     perform_full_analysis,
                                     FUZZY_MATCH_CUTOFF_F500,
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
                                     fairr_deforestation_score_threshold_mediocre_or_weak)

"Add score via scoring function"

# Weight dictionary for the weighted average approach
weight_dictionary = {'human_rights_policy_in_place': 0.2,
                     'deforestation_policy_in_place': 0.2,
                     'forest500_score': 0.2,
                     'strength_hr_policy': 0.2,
                     'strength_defo_policy': 0.2
                     }

# Apply function to get dt2 scores and weighted average
dt2_scores = apply_dt2_weighted_average_approach(df_portfolio, weight_dictionary, cutoffs=(0.9, 0.7))
df_portfolio = df_portfolio.merge(dt2_scores, on='identifier', how='left')

# EXPORT
df_portfolio.to_excel(os.path.join(PATH_TO_OUTPUT_FOLDER,
                                   'output_final_dt2.xlsx'), sheet_name='output_final_dt2', index=False)
