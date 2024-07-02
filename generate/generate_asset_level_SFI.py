"""
Description:
    This script collects and prepares asset level from the Spatial Finance Initiative (SFI).
    Link here: https://www.cgfi.ac.uk/spatial-finance-initiative/geoasset-project/ (which needs to be checked for updates)

Update:
    Last updated in November 2023

Output:
    A cleaned and structured DataFrame saved as a CSV file
    
NOTES:
- There are 6952 rows in the combined dataframe
- There are 122 rows with a parent_name_2 (i.e. more than one parent). This could still be incorporated.
- From columns like production_type, plant_type, etc., one could perhaps derive more accurate sector information 
    if needed.
"""

# import 
import os
import pandas as pd
from filepaths import PATH_TO_INPUT_FOLDER


# define function to prepare SFI data

def process_and_save_sfi_data():
    """
    Collect and prepare data from the Spatial Finance Initiative (SFI).

    This function processes data from various asset-level datasets (e.g., steel, cement, pulp and paper, petrochemicals,
    wastewater, beef) provided by the Spatial Finance Initiative. The steps include:
    1. Loading and cleaning data.
    2. Filtering for relevant data points.
    3. Standardizing column names and units.
    4. Consolidating data into a single DataFrame.
    5. Adding additional information such as sector categorization and unique identifiers.

    The function also includes helper functions for cleaning column names and converting columns to numeric values.

    Returns:
        pd.DataFrame: A consolidated and cleaned DataFrame containing asset-level data from the SFI.
    """
    ##################################################
    ######    0) HELPER FUNCTION                ######
    ##################################################

    # function that turns all columns into lowercase and replaces spaces with underscores
    def clean_col_names(df):
        df.columns = df.columns.str.lower()
        df.columns = df.columns.str.replace(" ", "_")
        return df

    # define input path
    input_path = os.path.join(PATH_TO_INPUT_FOLDER,
                              'asset_level_data/spatial_finance_initiative', "SFI_data_preprocessed.xlsx")

    # Function to convert the column to numeric
    def convert_to_numeric(df, column_name):
        # Strip whitespace and convert 'Not applicable' and 'No information' to NaN
        df[column_name] = pd.to_numeric(
            df[column_name].str.strip().replace(["Not applicable", "No information"], [pd.NA, pd.NA]), errors='coerce')
        return df

    # define relevant columns
    vars_to_keep = ['uid', 'city', 'state', 'country', 'iso3', 'latitude', 'longitude',
                    'status', 'owner_permid', 'owner_name', 'owner_lei', 'parent_permid',
                    'parent_name', 'parent_lei', 'parent_ticker', 'parent_exchange', 'capacity', 'capacity_unit',
                    'sector']

    ##################################################
    ######    1) DATA IMPORT AND PREPARATION    ######
    ##################################################

    ### --- 1.1) STEEL --- ###

    df_steel = pd.read_excel(input_path, sheet_name="steel")
    df_steel = clean_col_names(df_steel)  # turn cols into lowercase

    # STATUS
    df_steel = df_steel[df_steel["status"] == "Operating"]

    # CAPACITY
    df_steel["capacity_unit"] = "millions of tons - steel/cement"

    # OWNERSHIP INFORMARTION

    columns_to_check = ['parent_lei', 'owner_lei', 'parent_permid', 'owner_permid']

    for column in columns_to_check:
        if column not in df_steel.columns:
            df_steel[column] = float('nan')

    # SECTOR
    df_steel['primary_product'].fillna('', inplace=True)
    df_steel["sector"] = "steel" + "/" + df_steel["primary_product"].str.lower()

    # EXPORT
    df_steel = df_steel[vars_to_keep]

    ### --- 1.2) CEMENT --- ###

    df_cement = pd.read_excel(input_path, sheet_name="cement")
    df_cement = clean_col_names(df_cement)  # turn cols into lowercase

    # STATUS
    df_cement = df_cement[df_cement["status"] == "Operating"]

    # CAPACITY
    df_cement["capacity_unit"] = "millions of tons - steel/cement"

    # OWNERSHIP INFORMARTION
    for column in columns_to_check:
        if column not in df_cement.columns:
            df_cement[column] = float('nan')

    # SECTOR
    df_cement['production_type'].fillna('', inplace=True)
    df_cement["sector"] = "cement" + "/" + df_cement["production_type"].str.lower()

    # EXPORT
    df_cement = df_cement[vars_to_keep]

    ### --- 1.3) PULP AND PAPER --- ###
    df_pulp_paper = pd.read_excel(input_path, sheet_name="pulp_paper")
    df_pulp_paper = clean_col_names(df_pulp_paper)  # turn cols into lowercase

    # STATUS
    df_pulp_paper = df_pulp_paper[df_pulp_paper["status"] == "operating"]

    # CAPACITY
    # capacity_pulp: Total pulp production capacity of the asset in tonnes
    # capacity_paper: Total paper production capacity of the asset in tonnes
    # Assumption: sometimes an asset produces both, here we assume they can be added together

    # Convert to string, strip whitespace, replace non-numeric values with 0, and sum capacities
    df_pulp_paper["capacity_pulp"] = pd.to_numeric(df_pulp_paper["capacity_pulp"].astype(str).str.strip(),
                                                   errors='coerce').fillna(0)
    df_pulp_paper["capacity_paper"] = pd.to_numeric(df_pulp_paper["capacity_paper"].astype(str).str.strip(),
                                                    errors='coerce').fillna(0)
    df_pulp_paper["capacity"] = df_pulp_paper["capacity_pulp"] + df_pulp_paper["capacity_paper"]

    df_pulp_paper["capacity_unit"] = "pulp and paper production capacity (tons)"

    # OWNERSHIP INFORMARTION
    for column in columns_to_check:
        if column not in df_pulp_paper.columns:
            df_pulp_paper[column] = float('nan')

    # SECTOR
    df_pulp_paper['planty_type'].fillna('', inplace=True)
    df_pulp_paper["sector"] = "pulp paper" + "/" + df_pulp_paper["planty_type"].str.lower()

    # EXPORT
    df_pulp_paper = df_pulp_paper[vars_to_keep]

    ### --- 1.4) PETROCHEMICALS --- ###
    df_petrochemicals = pd.read_excel(input_path, sheet_name="petrochemicals")
    df_petrochemicals = clean_col_names(df_petrochemicals)  # turn cols into lowercase

    # STATUS
    df_petrochemicals = df_petrochemicals[df_petrochemicals["status"] == "Operating"]

    # CAPACITY
    df_petrochemicals["capacity_unit"] = "chemical production capacity (thousands of metric tons)"

    # OWNERSHIP INFORMARTION
    for column in columns_to_check:
        if column not in df_petrochemicals.columns:
            df_petrochemicals[column] = float('nan')

    # SECTOR
    df_petrochemicals['petrochemical'].fillna('', inplace=True)
    df_petrochemicals["sector"] = "petrochemicals" + "/" + df_petrochemicals["petrochemical"]

    # EXPORT
    df_petrochemicals = df_petrochemicals[vars_to_keep]

    ### --- 1.5) WASTEWATER --- ###
    df_wastewater = pd.read_excel(input_path, sheet_name="wastewater")
    df_wastewater = clean_col_names(df_wastewater)  # turn cols into lowercase

    # STATUS
    df_wastewater = df_wastewater[df_wastewater["status"] == "active"]

    # CAPACITY
    df_wastewater["capacity_unit"] = "wastewater treatment capacity in population equivalent"
    # NOTE perhaps there is even more accurate information in load_entering

    # OWNERSHIP INFORMARTION
    for column in columns_to_check:
        if column not in df_wastewater.columns:
            df_wastewater[column] = float('nan')

    # SECTOR
    df_wastewater['primary_treatment'].fillna('', inplace=True)
    df_wastewater["sector"] = "wastewater" + "/" + df_wastewater["primary_treatment"].str.lower()

    # EXPORT
    df_wastewater = df_wastewater[vars_to_keep]

    ### --- 1.6) BEEF --- ###
    df_beef = pd.read_excel(input_path, sheet_name="beef")
    df_beef = clean_col_names(df_beef)  # turn cols into lowercase

    # STATUS
    df_beef = df_beef[df_beef["status"] == "active"]

    # CAPACITY
    df_beef.rename(columns={"capacity_annually": "capacity"}, inplace=True)
    df_beef["capacity_unit"] = "beef slaughtering capacity (heads/year)"

    # OWNERSHIP INFORMARTION
    for column in columns_to_check:
        if column not in df_beef.columns:
            df_beef[column] = float('nan')

    # SECTOR
    df_beef['facility_type'].fillna('', inplace=True)
    df_beef["sector"] = "beef" + "/" + df_beef["facility_type"].str.lower()

    # EXPORT
    df_beef = df_beef[vars_to_keep]

    ##################################################
    ######    2) CONSOLIDATION                  ######
    ##################################################

    # combine all dataframes
    list_of_dfs = [df_steel, df_cement, df_pulp_paper, df_petrochemicals, df_wastewater, df_beef]
    df_sfi = pd.concat(list_of_dfs)

    # turn owner_name and parent_name into strings
    string_variables = ['owner_name', 'parent_name']
    for var in string_variables:
        df_sfi[var] = df_sfi[var].astype(str)
        df_sfi[var].fillna('', inplace=True)  # fill nan with empty string

    # Rename columns
    df_sfi.rename(columns={'iso3': 'country_iso'}, inplace=True)

    # Generate a unique identifier
    df_sfi['uid'] = ['SFI_' + str(num) for num in list(range(len(df_sfi)))]

    return df_sfi
