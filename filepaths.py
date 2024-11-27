'''
This file compiles all different file paths that are used in the project. 
It is used to keep track of all the different file paths and to make it easier to change them if necessary. 

Note that the PATH_TO_REPO variable needs to be changed in your python project folder path!
'''
from pathlib import Path

# All relative file paths are defined here:

# Define the base path
PATH_TO_REPO = Path('/Users/lokeshpandey/Library/CloudStorage/OneDrive-stud.uni-goettingen.de/Clim_Com')

import sys


sys.path.append('PATH_TO_REPO')

# Define input and output folders
PATH_TO_OUTPUT_FOLDER = PATH_TO_REPO / 'data/output'
PATH_TO_INPUT_FOLDER = PATH_TO_REPO / 'data/input'
PATH_TO_FIGURES_FOLDER = PATH_TO_OUTPUT_FOLDER / 'figures'

# Portfolio data
PATH_TO_PORTFOLIO_DATA = PATH_TO_INPUT_FOLDER / 'portfolio_data.xlsx'

# Define file paths of final results
PATH_TO_DT1 = PATH_TO_OUTPUT_FOLDER / 'output_final_dt1.xlsx'
PATH_TO_DT2 = PATH_TO_OUTPUT_FOLDER / 'output_final_dt2.xlsx'

# Classification mappings
PATH_TO_NACE_GICS_MAPPING = PATH_TO_INPUT_FOLDER / 'classifications_mapping/GICS_NACE_crosswalk.xlsx'
PATH_TO_NACE_FAO_MAPPING = PATH_TO_INPUT_FOLDER / 'classifications_mapping/nace_to_fao.xlsx'
PATH_TO_DEDUCE_MAPPING = PATH_TO_INPUT_FOLDER / 'classifications_mapping/deduce_lookup.xlsx'

# Literature analysis high impact sectors
PATH_TO_SECTOR_FLAGS = PATH_TO_INPUT_FOLDER / 'sector_flags_direct_indirect.xlsx'

# Asset level data
PATH_TO_COMPANY_SPECIFIC_ASSET_LEVEL_DATA = PATH_TO_OUTPUT_FOLDER / 'internal_data/asset_level_data_aggregated.csv'
PATH_TO_DISAGGREGATED_ASSETS = PATH_TO_OUTPUT_FOLDER / 'internal_data/asset_level_data_disaggregated.csv'
PATH_CLIMATE_TRACE_IDENTIFIERS = PATH_TO_INPUT_FOLDER / 'asset_level_data/climate_trace/ownership-climate-trace_entity-org-id_113023.csv'
PATH_CLIMATE_TRACE_IDENTIFIERS_SOURCE = PATH_TO_INPUT_FOLDER / 'asset_level_data/climate_trace/ownership-climate-trace_org-id_113023.csv'

# Forest finance dataset
PATH_TO_FOREST_FINANCE = PATH_TO_INPUT_FOLDER / 'forest&finance_dataset_nov23.xlsx'

# Data linked to direct attribution
# to download/update faostat data: https://www.fao.org/faostat/en/#data/QV
PATH_TO_FAOSTAT_2018 = PATH_TO_INPUT_FOLDER / 'direct_deforestation_attribution/FAOSTAT_2018.xls'
# to download/update faostat production data: https://www.fao.org/faostat/en/#data/QCL
PATH_TO_FAOSTAT_PRODUCTION_2018 = PATH_TO_INPUT_FOLDER / 'direct_deforestation_attribution/FAOSTAT_PRODUCTION_2018.xls'
# to download/update faostat production price data: https://www.fao.org/faostat/en/#data/PP
PATH_TO_FAOSTAT_PRODUCTION_PRICE_2018 = PATH_TO_INPUT_FOLDER / 'direct_deforestation_attribution/FAOSTAT_PRODUCTION_PRICE_2018.xls'
PATH_TO_GDP_DATA_2018 = PATH_TO_INPUT_FOLDER / 'direct_deforestation_attribution/GDP_data_2018.xlsx'
PATH_TO_UPDATED_DEFORESTATION_ATTRIBUTION_DATA = PATH_TO_INPUT_FOLDER / 'direct_deforestation_attribution/direct_attribution_data.xlsx'

# Data relevant for the hotspot analysis
PATH_TO_DEFORESTATION_HOTSPOTS_2023 = PATH_TO_INPUT_FOLDER / 'geospatial_data/Emerging_Hot_Spots_2023.geojson'
PATH_TO_SUBSIDIARY_LOCATIONS = PATH_TO_INPUT_FOLDER / '...'  # NEEDS TO BE FILLED IN

# Incorporation of Forest500 data
PATH_TO_F500_FIRMS = PATH_TO_INPUT_FOLDER / 'forest500_data/forest500-companies-data-download-2023.csv'
PATH_TO_F500_FIS = PATH_TO_INPUT_FOLDER / 'forest500_data/forest500-institutions-data-download-2023.csv'
PATH_TO_F500_MANUAL_MATCHES = PATH_TO_INPUT_FOLDER / 'forest500_data/forest500_2023_manual_matches.csv'

# Open source policy data
PATH_TO_POLICY_DATA = PATH_TO_INPUT_FOLDER / 'policy_data'
PATH_TO_WBA = PATH_TO_POLICY_DATA / 'wba_2023_nature_benchmark_June2024.xlsx'
PATH_TO_SBTI = PATH_TO_POLICY_DATA / 'sbti_companies_June2024.xlsx'

PATH_TO_FOOD_EMISSIONS_50 = PATH_TO_POLICY_DATA / 'food_emissions_50_benchmark.xlsx'
PATH_TO_SPOTT_PALM_OIL = PATH_TO_POLICY_DATA / 'spott_palm_oil_companies.csv'
PATH_TO_SPOTT_RUBBER = PATH_TO_POLICY_DATA / 'spott_natural_rubber.csv'
PATH_TO_SPOTT_TIMBER_ETC = PATH_TO_POLICY_DATA / 'spott_timber_pulp_paper.csv'
PATH_TO_SPOTT_PALM_OIL_QUESTIONS = PATH_TO_POLICY_DATA / 'SPOTT-palm-oil-companies-assessment-data-downloaded-2024-05-23.csv'
PATH_TO_SPOTT_TIMBER_ETC_QUESTIONS = PATH_TO_POLICY_DATA / 'SPOTT-timber-pulp-paper-companies-assessment-data-downloaded-2024-05-28.csv'
PATH_TO_SPOTT_RUBBER_QUESTIONS = PATH_TO_POLICY_DATA / 'SPOTT-natural-rubber-companies-assessment-data-downloaded-2024-05-28.csv'
PATH_TO_DEFORESTATION_ACTION_TRACKER = PATH_TO_POLICY_DATA / '2023_Deforestation_Action_Tracker_complete_dataset_Nov23.xlsx'

PATH_TO_VARIABLE_DESCRIPTIONS = PATH_TO_INPUT_FOLDER / 'final_excel_variable_descriptions.xlsx'

# Data that can be incorporated!
PATH_TO_FAIRR = PATH_TO_POLICY_DATA / 'FAIRR_data_PLACEHOLDER.xlsx'  # note to user: insert file name if access
PATH_TO_CDP_DATA = PATH_TO_POLICY_DATA / 'CDP_forests_data_PLACEHOLDER.xlsx'  # note to user: insert file name if access
