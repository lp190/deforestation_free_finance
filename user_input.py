'''
user_input.py file

This file stores all the variables etc with assumptions that could be changed by the user.

Structure:
    1) USER INPUT (parameters to be set by the user)
    2) DEVELOPER INPUT (parameterswe don't suggest to change)
'''

# ------------------------- #
# --- 1) USER INPUT ------- #
# ------------------------- #

#### Define parameters for the decision tree 1
'''
INSTRUCTIONS:
     
    CHOSE EACH PARAMETER:
    "exclude_financial_institutions"
        -> If True then all FIs are excluded from the portfolio before running the analysis
        -> If no loan data is available the deforestation exposure proxies are less credibly for FIs 
    "perform_full_analysis"
        -> if True it will actually run the filters for the variable values chosen by the user and the analysis will be
           carried out from scratch - BE AWARE THE OUTPUT FOLDER WILL BE DELETED
        -> if False, will allow the use of intermediary data files to reduce time, but can be misleading because if the
           user changed a variable value in between it will not be reflected in the results (since intermediary files
           are loaded)
    
    CHOSE ONE BIAS TURE AND ONE FALSE:
    "bias_towards_existing_data"
        -> If true then certain data sources which do not offer data for certain variables/indicators for stocks in your
           portfolio are treated as falsely NA (i.e. if there are no assets in Brazil then we assume that this is not
           because of missing data but that there are actually no assets in Brazil linked to this stock) 
    "bias_towards_missing_data"
        -> If true than certain data sources which do not offer data for certain variables/indicators for stocks in your
           portfolio are treated as truly NA (i.e. if there are no assets in Brazil then we assume that this is purely
           because of missing data hence we assign a zero instead of NA to this region) 
    
    CHOSE ONE WEIGHTING TRUE AND ALL THE OTHERS FALSE:
    "equal_weighting" 
        -> Weights are distributed equally amongst datasource
    "climate_and_company_weighting"
        -> Based on our expertise when cleaning and compiling the data we suggest a weighted approach
        ->...
    "specific_weighting"
        -> Based on your expertise and aims please specify the detailed weighting that should be applied
        -> ...
'''

# General parameters (CHOSE EACH PARAMETER)
exclude_financial_institutions = False
# NB: note this does not regenerate the files that are precomputed by the "generate()" functions
# (supply chain data, direct attribution data, etc). If you want to recompute those, please manually remove.
perform_full_analysis = True

# Parameters on data source compilation (SET ONE BIAS TO TRUE AND ONE TO FALSE)
bias_towards_existing_data = True
bias_towards_missing_data = False

# Parameters on data source weighting (CHOSE ONE WEIGHTING TRUE AND ALL THE OTHERS FALSE)
equal_information_importance = False
climate_and_company_information_importance = True
specific_information_importance = False

if equal_information_importance:
    importance_revenue_info = 1
    importance_hierarchy_info = 1
    importance_asset_info = 1
    importance_headquarter_info = 1
elif climate_and_company_information_importance:
    importance_revenue_info = 3
    importance_hierarchy_info = 2
    importance_asset_info = 2
    importance_headquarter_info = 1
# If you set specific_weighting = True then please specify your weightings below
elif specific_information_importance:
    importance_revenue_info = 0
    importance_hierarchy_info = 0
    importance_asset_info = 0
    importance_headquarter_info = 0

### APPLY DECISION TREE 1

# Main function
use_io_model_score = True
use_trase_flag = False  # not part of online repository
use_flag_direct = True
use_flag_indirect = True
use_flag_forest500 = True
flag_direct_threshold_high = 0.75
flag_direct_threshold_medium = 0.5
flag_indirect_threshold_high = 0.75
flag_indirect_threshold_medium = 0.5
cutoff_direct_attribution = 0
IO_threshold_high = 0.1
IO_threshold_medium = 0.04
hotspot_assets_threshold = 1
hotspot_subsidiaries_threshold = 1
asset_data_exists = True
subsidiary_data_exists = False

# The following only work if proprietary ESG data is incorporated
recent_controversies_cutoffs = False  # only works if proprietary ESG data is incorporated
historical_controversies_cutoffs = False  # only works if proprietary ESG data is incorporated
total_controversies_cutoffs = False  # only works if proprietary ESG data is incorporated
recent_controversies_threshold_high = 2.0
recent_controversies_threshold_medium = 1.0
historical_controversies_threshold_high = 3.0
historical_controversies_threshold_medium = 2.0

### APPLY DECISION TREE 2

## General Input
# if you integrated CDP FOrests data or proprietary data, set the following to True
cdp_data_exists = False
proprietary_policy_data_exists = False

## STEP 1
# Define based on Forest500 score whether companies have sufficient deforestation and human rights policies in place
forest500_threshold_bucketing_dt2 = 60

## STEP 2 - No user input needed, the apply_decision_tree2_logic.py module checks whether there is any indication that
##          human rights and deforestation policy exists

## STEP 3 - human rights and deforestation policy strength
# Define thresholds for companies with strong human rights policies
refinitiv_hr_score_threshold_strong = 0.8
spott_community_land_labour_score_threshold_strong = 80
spott_smallholders_suppliers_score_threshold_strong = 80
spott_gov_grievance_score_threshold_strong = 80
wba_ma3_social_inclusion_and_community_impact_threshold_strong = 50
fairr_work_score_threshold_strong = 60
# Define thresholds for companies with mediocre (if >=) or weak (if <) human rights policies
refinitiv_hr_score_threshold_mediocre_or_weak = 0.4
spott_community_land_labour_score_threshold_mediocre_or_weak = 40
spott_smallholders_suppliers_score_threshold_mediocre_or_weak = 40
spott_gov_grievance_score_threshold_mediocre_or_weak = 40
wba_ma3_social_inclusion_and_community_impact_threshold_mediocre_or_weak = 20
fairr_work_score_threshold_mediocre_or_weak = 40
# Define thresholds for companies with strong deforestation policies
spott_def_biodiv_score_threshold_strong = 80
wba_ma2_ecosystems_and_biodiversity_threshold_strong = 25
dat_score_threshold_strong = 25
fairr_deforestation_score_threshold_strong = 60
# Define thresholds for companies with mediocre (if >=) or weak (if <) deforestation policies
spott_def_biodiv_score_threshold_mediocre_or_weak = 40
wba_ma2_ecosystems_and_biodiversity_threshold_mediocre_or_weak = 10
dat_score_threshold_mediocre_or_weak = 10
fairr_deforestation_score_threshold_mediocre_or_weak = 40

# ------------------------- #
# --- 2) DEVELOPER INPUT ------- #
# ------------------------- #


### REFINITIV
QUALITY_THRESHOLD_SECTORAL_REVENUE = 0.8
QUALITY_THRESHOLD_GEOGRAPHICAL_REVENUE = 0.6

## FOREST AND FINANCE DATA
FOREST_FINANCE_YOI = 2023
FOREST_FINANCE_TRUNCATION = 5
# in millions of dollars (note: this needs to be adapted if the number of years of truncation is changed)
# in the case of FOREST_FINANCE_THRESHOLD_SUM=1.0 and FOREST_FINANCE_TRUNCATION=5, about ~1K of the ~3K FIs are flagged
FOREST_FINANCE_THRESHOLD = 1.0

# FILE: prep_orbis_hierarchy_data_download_compilation
RELEVANCE_THRESHOLD_SUBSIDIARY = 50.01
EXCLUDE_POTENTIALLY_UNIMPORTANT_SUBSIDIARIES = 1

# FILE: apply_controversy.py
FUZZY_MATCH_CUTOFF_F500 = 85  # this is also used in DT2 in order to perform a full analysis
THRESHOLD_ESG_CONTROVERSY_SCORE_TO_FLAG = 25

# Thresholds within the function

### FILE: apply direct attribution
# NOTE! 2022 is bad data quality, so last year to be used is 2021
# NOTE2! for now: only 2018 can be used as it depends on FAOSTAT data which has been processed only for 2018
DIRECT_ATTRIBUTION_YEAR = 2018
if not DIRECT_ATTRIBUTION_YEAR == 2018:
    raise NotImplementedError('please adapt appropriately in the generate_direct_attribution_filter.py '
                              'file as well as the required underlying datasets')

# note that this only changes which attribution data gets used for the direct filter, not for the IO model
USE_AMORTIZED = True

# in cases where FAOSTAT data is missing, and there are no valid group averages that can be taken, you can specify
# for the size of the sector region pair to be approximated as a fixed percentage of the country's GDP for that year
# 919 is leather; let's assume 0.25% of the total GDP
# 6716 is forest plantation; let's assume 0.5% of the total GDP
GDP_PROPORTIONS_DICT = {919: 0.0025, 6716: 0.005}
# in FAOSTAT, the cattle sector, for many countries is comprised of multiple (indigenous) species
# assumption: include the different cattle meat sectors and buffalo meat sectors, but exclude pigs, goats, sheep etc.
# note that also buffalo (947, 972, 973, 948) could reasonably be excluded from this list
CATTLE_SECTORS = [866, 867, 868, 944, 945, 870, 947, 972, 973, 948]

# Threshold for deforestation attribution; entries with deforestation risk below this value are filtered out.
DIRECT_ATTRIBUTION_HECTARE_CUTOFF = 0.0001

## FILE: apply_open_source_score.py
STRING_MATCHING_THRESHOLD_OPEN_SOURCE = 90

## DEFORESTATION HOTSPOTS
DISTANCE_THRESHOLD_ASSETS = 50
DISTANCE_THRESHOLD_SUBSIDIARIES = 50

# Dictionary to define the weights for various types of hotspots
IMPACT_DICT = {
    0: 1.0,  # Diminishing Hot Spot
    1: 3.0,  # Intensifying Hot Spot
    2: 2.0,  # New Hot Spot
    3: 3.0,  # Persistent Hot Spot
    4: 1.0  # Sporadic Hot Spot
}

## DEFORESTATION ACTION TRACKER
DAT_THRESHOLD = 20.0

NEW_TO_OLD_GICS = dict(zip(
    [60201010, 60202020, 60202030, 60202040, 20202030, 60104010, 60102510, 60107010, 25503030, 60108040,
     60105010, 40201060, 60108010, 60201030, 60201020, 20304040, 60106010, 60201040, 20304030, 60106020,
     60108030, 60108050, 60108020, 60103010],
    [60102010, 60102020, 60102030, 60102040, 45102020, 60101040, 60101020, 60101070, 25203010, 60101080,
     60101050, 40201040, 60101080, 60102030, 60102020, 20304020, 60101060, 60102040, 20304020, 60101060,
     60101080, 60101080, 60101080, 60101030]
))

EXIOBASE3_YEAR = 2018

# NOTE! 2022 is bad data quality, so last year to be used is 2021
IO_DEFORESTATION_ATTRIBUTION_DATA_YEAR = 2021

columns_to_load_eikon = [
    'ric',
    'isin',
    'name',
    'country_domicil_name',
    'var_impact_controv',
    'var_impact_controv_count',
    'var_impact_controv_recent',
    'var_esg_controv_sco']

region_mapping_ISO3_EXIO = {'AFG': 'WA', 'ALB': 'WE', 'DZA': 'WF', 'ASM': 'WA', 'AND': 'WE', 'AGO': 'WF', 'AIA': 'WL',
                            'ATA': 'WA', 'ATG': 'WL', 'ARG': 'WL', 'ARM': 'WA', 'ABW': 'WL', 'AUS': 'AU', 'AUT': 'AT',
                            'AZE': 'WA', 'BHS': 'WL', 'BHR': 'WM', 'BGD': 'WA', 'BRB': 'WL', 'BLR': 'WE', 'BEL': 'BE',
                            'BLZ': 'WL', 'BEN': 'WF', 'BMU': 'WL', 'BTN': 'WA', 'BOL': 'WL', 'BES': 'WL', 'BIH': 'WE',
                            'BWA': 'WF', 'BVT': 'WA', 'BRA': 'BR', 'BA1': 'WA', 'IOT': 'WA', 'BRN': 'WA', 'BGR': 'BG',
                            'BFA': 'WF', 'BDI': 'WF', 'KHM': 'WA', 'CMR': 'WF', 'CAN': 'CA', 'CPV': 'WF', 'CYM': 'WL',
                            'CAF': 'WF', 'TCD': 'WF', 'CHI': 'WE', 'CHL': 'WL', 'CHN': 'CN', 'CXR': 'WA', 'CCK': 'WA',
                            'COL': 'WL', 'COM': 'WF', 'COD': 'WF', 'COG': 'WF', 'COK': 'WA', 'CRI': 'WL', 'CIV': 'WF',
                            'HRV': 'HR', 'CUB': 'WL', 'CUW': 'WL', 'CYP': 'CY', 'CZE': 'CZ', 'DNK': 'DK', 'DJI': 'WF',
                            'DMA': 'WL', 'DOM': 'WL', 'ECU': 'WL', 'EGY': 'WM', 'SLV': 'WL', 'GNQ': 'WF', 'ERI': 'WF',
                            'EST': 'EE', 'ETH': 'WF', 'FRO': 'WE', 'FLK': 'WL', 'FJI': 'WA', 'FIN': 'FI', 'FRA': 'FR',
                            'GUF': 'WL', 'PYF': 'WA', 'GAB': 'WF', 'GMB': 'WF', 'GEO': 'WA', 'DEU': 'DE', 'GHA': 'WF',
                            'GIB': 'WE', 'GRC': 'GR', 'GRL': 'WL', 'GRD': 'WL', 'GLP': 'WL', 'GUM': 'WA', 'GTM': 'WL',
                            'GIN': 'WF', 'GNB': 'WF', 'GUY': 'WL', 'HTI': 'WL', 'HMD': 'WA', 'HND': 'WL', 'HKG': 'WA',
                            'HUN': 'HU', 'ISL': 'WE', 'IND': 'IN', 'IDN': 'ID', 'IRN': 'WM', 'IRQ': 'WM', 'IRL': 'IE',
                            'IMY': 'WE', 'ISR': 'WM', 'ITA': 'IT', 'JAM': 'WL', 'JPN': 'JP', 'JOR': 'WM', 'KAZ': 'WA',
                            'KEN': 'WF', 'KIR': 'WA', 'PRK': 'WA', 'KSV': 'WE', 'KWT': 'WM', 'KGZ': 'WA', 'LAO': 'WA',
                            'LVA': 'LV', 'LBN': 'WM', 'LSO': 'WF', 'LBR': 'WF', 'LBY': 'WF', 'LIE': 'WE', 'LTU': 'LT',
                            'LUX': 'LU', 'MAC': 'WA', 'MKD': 'WE', 'MDG': 'WF', 'MWI': 'WF', 'MYS': 'WA', 'MDV': 'WA',
                            'MLI': 'WF', 'MLT': 'MT', 'MHL': 'WA', 'MTQ': 'WL', 'MRT': 'WF', 'MUS': 'WF', 'MYT': 'WF',
                            'MEX': 'MX', 'FSM': 'WA', 'MDA': 'WE', 'MCO': 'WE', 'MNG': 'WA', 'MNE': 'WE', 'MSR': 'WL',
                            'MAR': 'WF', 'MOZ': 'WF', 'MMR': 'WA', 'NAM': 'WF', 'NRU': 'WA', 'NPL': 'WA', 'NLD': 'NL',
                            'ANT': 'WL', 'NCL': 'WA', 'NZL': 'WA', 'NIC': 'WL', 'NER': 'WF', 'NGA': 'WF', 'NIU': 'WA',
                            'NFK': 'WA', 'MNP': 'WA', 'NOR': 'NO', 'OMN': 'WM', 'PAK': 'WA', 'PLW': 'WA', 'PAL': 'WM',
                            'PAN': 'WL', 'PNG': 'WA', 'PRY': 'WL', 'PER': 'WL', 'PHL': 'WA', 'PCN': 'WA', 'POL': 'PL',
                            'PRT': 'PT', 'PRI': 'WL', 'QAT': 'WM', 'REU': 'WF', 'ROM': 'RO', 'RUS': 'RU', 'RWA': 'WF',
                            'WSM': 'WA', 'SMR': 'WE', 'STP': 'WF', 'SAU': 'WM', 'SEN': 'WF', 'SRB': 'WE', 'SYC': 'WF',
                            'SLE': 'WF', 'SGP': 'WA', 'SXM': 'WL', 'SVK': 'SK', 'SVN': 'SI', 'SLB': 'WA', 'SOM': 'WF',
                            'ZAF': 'ZA', 'SGS': 'WA', 'KOR': 'KR', 'SSD': 'WF', 'ESP': 'ES', 'LKA': 'WA', 'SHN': 'WF',
                            'KNA': 'WL', 'LCA': 'WL', 'SPM': 'WL', 'VCT': 'WL', 'SDN': 'WF', 'SUR': 'WL', 'SJM': 'WE',
                            'SWZ': 'WF', 'SWE': 'SE', 'CHE': 'CH', 'SYR': 'WM', 'TWN': 'TW', 'TJK': 'WA', 'EAT': 'WF',
                            'TZA': 'WF', 'THA': 'WA', 'TLS': 'WA', 'TGO': 'WF', 'TKL': 'WA', 'TON': 'WA', 'TTO': 'WL',
                            'TUN': 'WF', 'TUR': 'TR', 'TKM': 'WA', 'TCA': 'WL', 'TUV': 'WA', 'UGA': 'WF', 'UKR': 'WE',
                            'ARE': 'WM', 'GBR': 'GB', 'USA': 'US', 'UMI': 'WA', 'URY': 'WL', 'UZB': 'WA', 'VUT': 'WA',
                            'VAT': 'WE', 'VEN': 'WL', 'VNM': 'WA', 'VGB': 'WL', 'VIR': 'WL', 'WLF': 'WA', 'ESH': 'WF',
                            'YEM': 'WM', 'ZMB': 'WF', 'EAZ': 'WF', 'ZWE': 'WF', 'SRB and MNE': 'WE', 'PSE': 'WM',
                            'SDN and SSD': 'WF', 'ROU': 'RO'}

region_mapping_ISO2_EXIO = {'AF': 'WA', 'AL': 'WE', 'DZ': 'WF', 'AS': 'WA', 'AD': 'WE', 'AO': 'WF', 'AI': 'WL',
                            'AQ': 'WA', 'AG': 'WL', 'AR': 'WL', 'AM': 'WA', 'AW': 'WL', 'AU': 'AU', 'AT': 'AT',
                            'AZ': 'WA', 'BS': 'WL', 'BH': 'WM', 'BD': 'WA', 'BB': 'WL', 'BY': 'WE', 'BE': 'BE',
                            'BZ': 'WL', 'BJ': 'WF', 'BM': 'WL', 'BT': 'WA', 'BO': 'WL', 'BA': 'WE',
                            'EU': 'WE', 'FAIL': 'WA', 'GB-SCT': 'WE', 'GG': 'WA',
                            'BW': 'WF', 'BV': 'WA', 'BR': 'BR', 'B1': 'WA', 'IO': 'WA', 'BN': 'WA', 'BG': 'BG',
                            'BF': 'WF', 'BI': 'WF', 'KH': 'WA', 'CM': 'WF', 'CA': 'CA', 'CV': 'WF', 'KY': 'WL',
                            'CF': 'WF', 'TD': 'WF', 'CL': 'WL', 'CN': 'CN', 'CX': 'WA', 'CC': 'WA', 'CO': 'WL',
                            'KM': 'WF', 'CD': 'WF', 'CG': 'WF', 'CK': 'WA', 'CR': 'WL', 'CI': 'WF', 'HR': 'HR',
                            'CU': 'WL', 'CW': 'WL', 'CY': 'CY', 'CZ': 'CZ', 'DK': 'DK', 'DJ': 'WF', 'DM': 'WL',
                            'DO': 'WL', 'EC': 'WL', 'EG': 'WM', 'SV': 'WL', 'GQ': 'WF', 'ER': 'WF', 'EE': 'EE',
                            'ET': 'WF', 'FO': 'WE', 'FK': 'WL', 'FJ': 'WA', 'FI': 'FI', 'FR': 'FR', 'GF': 'WL',
                            'PF': 'WA', 'GA': 'WF', 'GM': 'WF', 'GE': 'WA', 'DE': 'DE', 'GH': 'WF', 'GI': 'WE',
                            'GR': 'GR', 'GL': 'WL', 'GD': 'WL', 'GP': 'WL', 'GU': 'WA', 'GT': 'WL', 'GN': 'WF',
                            'GW': 'WF', 'GY': 'WL', 'HT': 'WL', 'HM': 'WA', 'HN': 'WL', 'HK': 'WA', 'HU': 'HU',
                            'IS': 'WE', 'IN': 'IN', 'ID': 'ID', 'IR': 'WM', 'IQ': 'WM', 'IE': 'IE', 'IM': 'WE',
                            'IL': 'WM', 'IT': 'IT', 'JM': 'WL', 'JP': 'JP', 'JO': 'WM', 'KZ': 'WA', 'KE': 'WF',
                            'KI': 'WA', 'KP': 'WA', 'KW': 'WM', 'KG': 'WA', 'LA': 'WA', 'LV': 'LV', 'LB': 'WM',
                            'LS': 'WF', 'LR': 'WF', 'LY': 'WF', 'LI': 'WE', 'LT': 'LT', 'LU': 'LU', 'MO': 'WA',
                            'MK': 'WE', 'MG': 'WF', 'MW': 'WF', 'MY': 'WA', 'MV': 'WA', 'ML': 'WF', 'MT': 'MT',
                            'MH': 'WA', 'MQ': 'WL', 'MR': 'WF', 'MU': 'WF', 'YT': 'WF', 'MX': 'MX', 'FM': 'WA',
                            'MD': 'WE', 'MC': 'WE', 'MN': 'WA', 'ME': 'WE', 'MS': 'WL', 'MA': 'WF', 'MZ': 'WF',
                            'MM': 'WA', 'NR': 'WA', 'NP': 'WA', 'NL': 'NL', 'AN': 'WL', 'NC': 'WA', 'NZ': 'WA',
                            'NI': 'WL', 'NE': 'WF', 'NG': 'WF', 'NU': 'WA', 'NF': 'WA', 'MP': 'WA', 'NO': 'NO',
                            'OM': 'WM', 'PK': 'WA', 'PW': 'WA', 'PS': 'WM', 'PA': 'WL', 'PG': 'WA', 'PY': 'WL',
                            'PE': 'WL', 'PH': 'WA', 'PN': 'WA', 'PL': 'PL', 'PT': 'PT', 'PR': 'WL', 'QA': 'WM',
                            'RE': 'WF', 'RO': 'RO', 'RU': 'RU', 'RW': 'WF', 'WS': 'WA', 'SM': 'WE', 'ST': 'WF',
                            'SA': 'WM', 'SN': 'WF', 'RS': 'WE', 'SC': 'WF', 'SL': 'WF', 'SG': 'WA', 'SK': 'SK',
                            'SI': 'SI', 'SB': 'WA', 'SO': 'WF', 'ZA': 'ZA', 'GS': 'WA', 'KR': 'KR', 'ES': 'ES',
                            'LK': 'WA', 'SH': 'WF', 'KN': 'WL', 'LC': 'WL', 'PM': 'WL', 'VC': 'WL', 'SD': 'WF',
                            'SR': 'WL', 'SJ': 'WE', 'SZ': 'WF', 'SE': 'SE', 'CH': 'CH', 'SY': 'WM', 'TW': 'TW',
                            'TJ': 'WA', 'TZ': 'WF', 'TH': 'WA', 'TL': 'WA', 'TG': 'WF', 'TK': 'WA', 'TO': 'WA',
                            'TT': 'WL', 'TN': 'WF', 'TR': 'TR', 'TM': 'WA', 'TC': 'WL', 'TV': 'WA', 'UG': 'WF',
                            'UA': 'WE', 'AE': 'WM', 'GB': 'GB', 'US': 'US', 'UM': 'WA', 'UY': 'WL', 'UZ': 'WA',
                            'VU': 'WA', 'VA': 'WE', 'VE': 'WL', 'VN': 'WA', 'VG': 'WL', 'VI': 'WL', 'WF': 'WA',
                            'EH': 'WF', 'YE': 'WM', 'ZM': 'WF', 'ZW': 'WF', 'SS': 'WF', 'SX': 'WL', 'NA': 'WF'}
