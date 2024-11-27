"""
Description:
    This code compiles all the datapoints linked to DT1.
    And creates suggestions for buckets.

Outline of code:
    0) LOAD & DEFINE MASTER DATA (Data input) & SET PARAMETERS
    1): Collect flags for Decision Tree 1
        1.1: Apply sectoral filters
        1.2: Apply Forest and Finance
        1.3: Apply Indirect Deforestation Exposure via IO Model
        1.4: Apply Direct Country-Sector Exposure
        1.5: Add Geolocation Data
        1.6: Apply Controversy Data
        1.7: Add the number of location-sector pairs used per underlying datasource
    2): "BUCKETING" a la Global Canopy (suggestion)
        2.1: Add buckets via function created in apply_decision_tree1_logic.py
        2.2: Add score via scoring function
    3): Export Output
    
OUTPUT
    - Excel file: output_final_dt1.xlsx
"""

# Import packages
import shutil
from pathlib import Path

import pandas as pd

from apply.apply_asset_level_hotspot import apply_deforestation_hotspots_assets
from apply.apply_controversy import apply_controversy_filters
from apply.apply_decision_tree1 import apply_dt1_conservative_approach, apply_dt1_weighted_average_approach
from apply.apply_direct_attribution import apply_direct_attribution_filter
from apply.apply_forest_and_finance import apply_forest_and_finance
from apply.apply_number_of_pairs import apply_number_of_pairs
from apply.apply_sectoral_filter_nace import apply_sectoral_filters_nace
from apply.apply_supply_chain_filter import apply_supply_chain_filter
# Import functions from other scripts
from filepaths import PATH_TO_PORTFOLIO_DATA, PATH_TO_COMPANY_SPECIFIC_ASSET_LEVEL_DATA, PATH_TO_OUTPUT_FOLDER, \
    PATH_TO_INPUT_FOLDER
from generate.generate_combine_asset_data import combine_asset_datasets
from prep.prep_asset_level_merge import merge_asset_level_to_portfolio_companies
from prep.prep_weighted_country_sector_pairs import prep_weighted_country_sector_pairs
from user_input import (
    # GENERAL SETTINGS
    exclude_financial_institutions,
    perform_full_analysis,
    bias_towards_existing_data,
    bias_towards_missing_data,
    equal_information_importance,
    climate_and_company_information_importance,
    specific_information_importance,
    importance_revenue_info,
    importance_hierarchy_info,
    importance_asset_info,
    importance_headquarter_info,
    # EXECUTE DT1
    use_io_model_score,
    use_trase_flag,
    use_flag_direct,
    use_flag_indirect,
    use_flag_forest500,
    recent_controversies_cutoffs,  # if true, absolute cutoffs are used
    historical_controversies_cutoffs,  # suggest to keep this false

    # EXECUTE BUCKETING
    flag_direct_threshold_high,
    flag_direct_threshold_medium,
    flag_indirect_threshold_high,
    flag_indirect_threshold_medium,
    IO_threshold_high,
    IO_threshold_medium,
    recent_controversies_threshold_high,
    recent_controversies_threshold_medium,
    historical_controversies_threshold_high,
    historical_controversies_threshold_medium,
    cutoff_direct_attribution,
    subsidiary_data_exists,
    hotspot_assets_threshold,
    hotspot_subsidiaries_threshold,

    # OVERLAY WITH HOTSPOTS
    DISTANCE_THRESHOLD_ASSETS
)
from utils import clean_df_portfolio

# from apply.apply_sectoral_filter_gics import apply_sectoral_filters_gics # not yet incorporated

### ---------------------------------------------------###
### 0) CORE DATA INPUT & SET PARAMETERS                ###
### ---------------------------------------------------###

"""
0.1: QUALITY CHECK PARAMETERS
"""
# Define lists of variables to check
boolean_variables = [
    exclude_financial_institutions, perform_full_analysis, bias_towards_existing_data,
    equal_information_importance, climate_and_company_information_importance, specific_information_importance,
    use_io_model_score, use_trase_flag, use_flag_direct, use_flag_indirect, use_flag_forest500,
    recent_controversies_cutoffs, historical_controversies_cutoffs
]

numerical_variables = [
    flag_direct_threshold_high, flag_direct_threshold_medium, flag_indirect_threshold_high,
    flag_indirect_threshold_medium, cutoff_direct_attribution, IO_threshold_high, IO_threshold_medium,
    recent_controversies_threshold_high, recent_controversies_threshold_medium,
    historical_controversies_threshold_high, historical_controversies_threshold_medium,
    hotspot_assets_threshold, hotspot_subsidiaries_threshold, importance_revenue_info, importance_hierarchy_info,
    importance_asset_info, importance_headquarter_info
]

# Perform boolean check
if any(not isinstance(variable, bool) for variable in boolean_variables):
    raise ValueError("All of the user inputs listed above must be boolean values (True or False), please check.")

# Perform numerical check
if any(not isinstance(variable, (int, float)) for variable in numerical_variables):
    raise ValueError("All of the user inputs listed above must be numerical values (integers or floats), please check.")

# Perform bias check
if bias_towards_existing_data == bias_towards_missing_data:
    raise ValueError('Please check your chosen bias and decide for one of them')

# Perform weighting check
if sum([
    equal_information_importance,
    climate_and_company_information_importance,
    specific_information_importance
]) >= 2:
    raise ValueError("Two or more weightings are set to True.")

"""
0.2: Delete interim files that were stored
"""

# If perform_full_analysis is set to True, delete the output folder and create a new one

if perform_full_analysis:
    shutil.rmtree(PATH_TO_OUTPUT_FOLDER, ignore_errors=True)
    PATH_TO_OUTPUT_FOLDER.mkdir()
    Path(PATH_TO_OUTPUT_FOLDER / 'internal_data').mkdir()

"""
0.3: LOAD DATA
"""

# Load df_portfolio (by default: MSCI ACWI Index with limited data)
df_portfolio = pd.read_excel(PATH_TO_PORTFOLIO_DATA, sheet_name='data').astype(str)
df_portfolio = clean_df_portfolio(df_portfolio)  # apply clean_df_portfolio from utils
df_portfolio["identifier"] = df_portfolio["permid"].astype(str)  # define identifier column

# Load asset level data (by default: data from SFI, CLimate Trace, GEM)
if Path(PATH_TO_COMPANY_SPECIFIC_ASSET_LEVEL_DATA).exists():
    asset_data_columns = ['permid', 'nace_code', 'country_iso', 'final_weight']
    df_asset_matches_aggregated = pd.read_csv(PATH_TO_COMPANY_SPECIFIC_ASSET_LEVEL_DATA,
                                              usecols=asset_data_columns,
                                              dtype={'permid': str, 'nace_code': str, 'country_iso': str,
                                                     'final_weight': float})
else:
    # Load the >70k assets from the different sources
    df_asset_data_raw = combine_asset_datasets()

    # Map to portfolio companies via identifiers & text matching
    df_asset_matches_disaggregated, df_asset_matches_aggregated = merge_asset_level_to_portfolio_companies(df_portfolio,
                                                                                                           df_asset_data_raw)

# Define main identifier column for asset data
df_asset_matches_aggregated["identifier"] = df_asset_matches_aggregated["permid"].astype(str)

# Derive country-sector pairs for each company in the portfolio
df_portfolio_country_sector_pairs = prep_weighted_country_sector_pairs(df_asset_matches_aggregated,
                                                                       df_portfolio,
                                                                       bias_towards_existing_data,
                                                                       bias_towards_missing_data,
                                                                       importance_asset_info,
                                                                       importance_headquarter_info)

### --------------------------------------------------------###
### 1) COLLECT FLAGS FOR DECISION TREE 1 (STEP1 of report)  ###
### --------------------------------------------------------###

"""
1.1: SECTORAL FILTERS
"""
# Apply sectoral filters to the portfolio data (Link to Report: Direct & Indirect Sectoral Flags)
df_portfolio = apply_sectoral_filters_nace(df_portfolio, df_portfolio_country_sector_pairs)

"""
1.2: FOREST AND FINANCE
"""
# Utilise the Forests & Finance data (Link to Report: Deforestation Exposure of Financial Institutions via Forest & Finance)
# Adjust sector column & respective codes to identify financial institutions

df_portfolio = apply_forest_and_finance(df_portfolio,
                                        sector_column="trbc_code_lev3",
                                        financial_sector_codes=["551010", "553010", "556010", "551020", "573010"])

"""
1.3: INDIRECT DEFORSTATION EXPOSURE via IO MODEL
"""
# Apply IO model to disaggregated portfolio data
df_portfolio = apply_supply_chain_filter(df_portfolio, df_portfolio_country_sector_pairs)

"""
1.4: DIRECT COUNTRY-SECTOR EXPOSURE 
"""
# Map the direct deforestation attribution data to country-sector pairs

df_portfolio = apply_direct_attribution_filter(df_portfolio, df_portfolio_country_sector_pairs)

"""
1.5: ADD GEOLOCATION DATA
"""
# Overlay corporate locations (via asset-level data) with deforestation hotspot data 

df_portfolio = apply_deforestation_hotspots_assets(df_portfolio,
                                                   type_of_asset='asset',
                                                   distance_threshold=DISTANCE_THRESHOLD_ASSETS)

# Overlay corporate locations (via subsidiary-level data) with deforestation hotspot data (IF AVAILABLE)

# df_portfolio = apply_deforestation_hotspots_assets(df_portfolio,
#                                                     type_of_asset='subsidiary',
#                                                     distance_threshold=DISTANCE_THRESHOLD_SUBSIDIARIES)

"""
1.6: APPLY CONTROVERSY DATA
"""
# Adds flags for forest500 (as a proxy for deforestation controversies) 
# A placeholder is included to incorporate proprietary ESG controversy data.

df_portfolio = apply_controversy_filters(df_portfolio, esg_controversies=False)

"""
1.7: ADD THE NUMBER OF LOCATION-SECTOR PAIRS USED PER UNDERLYING DATASOURCE
"""
# Adds the number of location-sector pairs used per underlying datasource

df_portfolio = apply_number_of_pairs(df_portfolio, df_portfolio_country_sector_pairs)

'''
Note on further datasets that could be included:
- Note that this is an open-source code repository, relying exclusively on publicly available data sources.
- If the user has access to proprietary data sources, they can be included in the analysis by following the same logic.
- I.e., by adding "flags" to the portfolio data, which can then be used to calculate the DT1 score.

Data points that have been removed from the public version of the code but are described in our report:
- Controversies: Recent and historical controversies (Link to Report: Controversies)
- Trase Earth / Trase Finance data
'''

### ---------------------------------------------------###
###  2): AGGREGATION (STEP 3 of report)                ###
### ---------------------------------------------------###

"""
This part is about aggregating the different flags.
See also the "Step 3 Chapter" in the Climate & Company et al. report.

Outline of code:
    2.1: Assign companies into buckets (low, medium, high, very high risk)
    2.2: Calculate Deforestation Exposure Score
"""

"""
2.1: Assign companies into buckets (low, medium, high, very high risk)
"""
# See function created in apply_decision_tree.py

df_portfolio = apply_dt1_conservative_approach(df_portfolio,
                                               use_io_model_score,
                                               use_trase_flag,
                                               use_flag_direct,
                                               use_flag_indirect,
                                               use_flag_forest500,
                                               recent_controversies_cutoffs,
                                               historical_controversies_cutoffs,
                                               flag_direct_threshold_high,
                                               flag_direct_threshold_medium,
                                               flag_indirect_threshold_high,
                                               flag_indirect_threshold_medium,
                                               IO_threshold_high,
                                               IO_threshold_medium,
                                               recent_controversies_threshold_high,
                                               recent_controversies_threshold_medium,
                                               historical_controversies_threshold_high,
                                               historical_controversies_threshold_medium,
                                               cutoff_direct_attribution,
                                               subsidiary_data_exists,
                                               hotspot_assets_threshold,
                                               hotspot_subsidiaries_threshold)

"""
2.2: Calculate Deforestation Exposure Score + Assign buckets via scoring function
"""

# Determine weights for companies and financial institutions.
# These weights can be adjusted by the user.

weight_dictionary_companies = {'flag_direct_score': 0.15,
                               'flag_indirect_score': 0.15,
                               'io_supply_chain_score': 0.30,
                               'flag_forest500': 0.1,
                               'direct_attribution_score': 0.15,
                               'asset_impact_assignment_count_asset': 0.15
                               }

weight_dictionary_financial_institutions = {'forest_and_finance_score': 0.30,
                                            'io_supply_chain_score': 0.30,
                                            'flag_forest500': 0.2,
                                            'direct_attribution_score': 0.1,
                                            'asset_impact_assignment_count_asset': 0.1
                                            }

# Apply function which creates a deforestation exposure score.
# The function also assigns a bucket based on the score.

df_portfolio = apply_dt1_weighted_average_approach(df_portfolio,
                                                   weight_dictionary_companies,
                                                   weight_dictionary_financial_institutions,
                                                   cutoffs=(0.9, 0.7))

### ---------------------------------------------------###
### 3)           EXPORT OUTPUT                         ###
### ---------------------------------------------------###

"""  
The remainder prepares the export into a clean .xlsx file.
We rename the columns and round the decimal variables.
The columns the user wants to keep etc. can be adjusted here.
"""

# Define list of basic firm-level variables to keep
basic_firm_level_variables = ["name", "identifier", "country_name", "country_iso", "nace_code", "nace_desc"]

# Define list of flags you want to keep
dt1_flags = ["flag_direct_score", "flag_indirect_score", "forest_and_finance_score",
             "forest_and_finance_amount", "io_supply_chain_score", "flag_forest500",
             "direct_attribution_score", "asset_impact_assignment_count_asset",
             "number_of_pairs_asset", "number_of_pairs_country_sector"]

dt1_aggregates = ["dt1_conservative", "dt1_bucket_based_on_score", "dt1_score", "dt1_score_winsorized"]
all_variables = basic_firm_level_variables + dt1_flags + dt1_aggregates

# Rename columns, subject to user's need
new_column_names = {'name': 'company_name',
                    "flag_direct_score": "sector_direct_score",
                    "flag_indirect_score": "sector_indirect_score",
                    "direct_attribution_score": "direct_attribution_score",
                    "IO_score": "IO_supply_chain_score",
                    "dt1_conservative": "dt1_bucket",
                    "dt1_bucket_based_on_score": "dt1_bucket_based_on_score",
                    "dt1_score_winsorized": "dt1_score_winsorized_and_normalized"}

output_final = df_portfolio[all_variables]
output_final.rename(columns=new_column_names, inplace=True)

# Define decimal variables & round them
decimal_vars = ['sector_direct_score', 'sector_indirect_score',
                'IO_supply_chain_score', 'dt1_score', 'dt1_score_winsorized']

output_final = output_final.round({col: 2 for col in decimal_vars})

# Store as excel file in sheet called "output_final"
output_final.to_excel(PATH_TO_OUTPUT_FOLDER / 'output_final_dt1.xlsx', sheet_name='output_final', index=False)
