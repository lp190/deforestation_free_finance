"""
Filename: prep_asset_level_open_source_merge.py

Description:
    Prepare all asset-level data, and derive the required columns for our analysis.
    
STRUCTURE:
    0) DATA PREP
    0.1) Import asset level data from .csv
    0.2) Helper functions
    1) MAPPING PROCESS
    1.1) Map on identifiers
    1.2) Map via direct text matching
    1.3) Map via fuzzy string matching
    1.4) Combine all matches
    2) COMBINE, ANALYSE AND CLEAN
    2.1) Check for potential asset duplicates and drop them
    2.2) Sector - NACE Mapping
    2.3) Weighting
    2.4) Derive ISO country codes
    3) EXPORT (and aggregate for IO modelling)
    

NOTES:
    - Cleaning company names function (mapping process) could be improved.
    - Some functions could be moved into a (separate) utils.py file to increase readibility.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pycountry
from rapidfuzz import fuzz, process

from filepaths import PATH_TO_OUTPUT_FOLDER
from utils import standardize_company_names, map_on_identifiers, merge_and_count_matches


def merge_asset_level_to_portfolio_companies(df_portfolio, df_asset_data):
    """
    Merges asset-level data with portfolio company data by performing several mapping processes.

    Parameters:
        df_portfolio (pd.DataFrame): DataFrame containing portfolio company data with columns 'name' and 'permid'.
        df_asset_data (pd.DataFrame): DataFrame containing asset-level data with various columns including identifiers and company names.

    Returns:
        tuple: A tuple containing two DataFrames:
            - df_asset_level_disaggregated (pd.DataFrame): Disaggregated asset-level data with essential columns.
            - df_asset_level_aggregated (pd.DataFrame): Aggregated asset-level data with unique permid-region-sector pairs and their final weights.
    """
    print("Merging asset-level data with portfolio company data...")

    # -----------------------------------------------
    # 0.1) DATA PREP
    # -----------------------------------------------

    ## STANDARDIZE COMPANY NAMES (via function from utils.py)
    df_portfolio['standardized_company_name'] = df_portfolio['name'].apply(standardize_company_names)
    df_asset_data['parent_name'] = df_asset_data['parent_name'].astype(str).apply(standardize_company_names)
    df_asset_data['owner_name'] = df_asset_data['owner_name'].astype(str).apply(standardize_company_names)
    df_asset_data['operator_name'] = df_asset_data['operator_name'].astype(str).apply(standardize_company_names)

    # Create a name & permid list for direct text matching
    df_portfolio_name2permid = df_portfolio.set_index('standardized_company_name')['permid'].to_dict()

    # Define a function that uses the mapping to find the permid for a given company name
    def get_permid_for_company(company_name, name_to_permid_mapping):
        """
        Retrieve the permid (Permanent Identifier) for a given company name using a provided mapping.

        This function looks up the permid for a specified company name from a dictionary mapping company names to permids.
        If the company name is not found in the mapping, the function returns None.

        Args:
            company_name (str): The name of the company for which to find the permid.
            name_to_permid_mapping (dict): A dictionary mapping company names (keys) to their corresponding permids (values).

        Returns:
            str or None: The permid for the specified company if found in the mapping; otherwise, None.
        """
        return name_to_permid_mapping.get(company_name, None)

    # -----------------------------------------------
    # 0.2) HELPER FUNCTIONS
    # -----------------------------------------------

    # see utils.py for the following functions:
    # - map_on_identifiers
    # - merge_and_count_matches

    ## Note that these two functions could also be moved to utils.py
    
    # FUZZY STRING MAPPING VIA RAPIDFUZZ

    def find_fuzzy(list_master, list_asset_data, score_cutoff: int, scorer=fuzz.ratio):
        """
        Finds fuzzy matches between two lists using rapidfuzz library.
        Read this article for more context: https://medium.com/@bewin4u/fuzzy-matching-for-million-row-address-dataset-with-rapidfuzz-and-splink-b704eaf1fda9

        Args:
            list_master (list): Our master data (i.e, our universe)
            list_asset_data (list): The list of strings to compare against the master list (example: df_climate_trace_unmatched["standardized_parent_name"].tolist()
            score_cutoff (int): The minimum score required for a match to be considered.
            scorer (function): The scoring function to use. Defaults to fuzz.ratio.

        Returns:
            list: A list of dictionaries containing the matched pairs and their scores.
        """
        # Generate a score matrix using rapidfuzz
        score_matrix = process.cdist(
            list_master,
            list_asset_data,
            processor=lambda x: str(x).lower(),  # Ensure strings are lowered
            scorer=scorer,
            dtype=np.uint8,  # Output the score as uint8, which is faster
            workers=-1,  # Use multithreading. -1 means use all cores
            score_cutoff=score_cutoff,
        )

        results = []
        # Find non-zero elements in the score matrix (indicating matches above the cutoff)
        master_indices, asset_data_indices = np.nonzero(score_matrix)
        for master_index, asset_data_index in zip(master_indices, asset_data_indices):
            results.append({
                "df_master_index": master_index,
                "asset_data_index": asset_data_index,
                "df_master_name": list_master[master_index],
                "asset_data_name": list_asset_data[asset_data_index],
                "score_of_match": score_matrix[master_index, asset_data_index],
            })
        return results

    # FUNCTION that builds on find_fuzzy to map on different columns and doing the data manipulation

    def map_fuzzy_matches(column_to_match, df_asset_unmatched, df_asset, df_master, score_cutoff):
        """
        Perform fuzzy matching between asset data and a master list to map permids (Permanent Identifiers).

        This function uses fuzzy matching to find the best matches between a specified column in the asset data and the
        standardized company names in a master list. It then maps the corresponding permids from the master list to the
        asset data based on the fuzzy match results and updates the asset data DataFrame with the mapped permids.

        Args:
            column_to_match (str): The name of the column in the asset data to be matched.
            df_asset_unmatched (pd.DataFrame): DataFrame containing the unmatched asset data.
            df_asset (pd.DataFrame): DataFrame containing the original asset data to be updated with matched permids.
            df_master (pd.DataFrame): DataFrame containing the master list with standardized company names and permids.
            score_cutoff (int): The score cutoff for fuzzy matching; only matches with a score above this cutoff are considered.

        Returns:
            pd.DataFrame: The original asset DataFrame updated with a new column containing the matched permids based on fuzzy matching.
        """
        # Perform fuzzy matching
        matches = find_fuzzy(list_master=df_master["standardized_company_name"].tolist(),
                             list_asset_data=df_asset_unmatched[column_to_match].tolist(),
                             score_cutoff=score_cutoff,
                             scorer=fuzz.ratio)

        # Create DataFrame from matches
        df_matches = pd.DataFrame(matches)

        # Map permid
        df_matches["matched_permid_fuzzy"] = df_master["permid"].iloc[df_matches["df_master_index"]].tolist()

        # Update original DataFrame
        df_matches.set_index('asset_data_index', inplace=True)
        df_asset[f"matched_permid_fuzzy_{column_to_match}"] = df_asset.index.map(df_matches['matched_permid_fuzzy'])

        # Print the name of df_asset
        print(
            f"matched_permid_fuzzy_{column_to_match} has been added to df_asset with a score cutoff of {score_cutoff}.")

        return df_asset

    # -----------------------------------------------
    # 1) MAPPING PROCESS
    # -----------------------------------------------

    #### 1.1) MAPPING BASED ON IDENTIFIERS
    print("Mapping based on identifiers...")

    # Map on identifiers, i.e. parent_lei, owner_lei, operator_lei, parent_permid, owner_permid, operator_permid

    df_matches_lei_parent = map_on_identifiers(df_asset_data, df_portfolio, 'parent_lei', 'lei')
    df_matches_lei_owner = map_on_identifiers(df_asset_data, df_portfolio, 'owner_lei', 'lei')
    df_matches_lei_operator = map_on_identifiers(df_asset_data, df_portfolio, 'operator_lei', 'lei')
    df_matches_permid_parent = map_on_identifiers(df_asset_data, df_portfolio, 'parent_permid', 'permid')
    df_matches_permid_owner = map_on_identifiers(df_asset_data, df_portfolio, 'owner_permid', 'permid')
    df_matches_permid_operator = map_on_identifiers(df_asset_data, df_portfolio, 'operator_permid', 'permid')

    # Create a list of dataframes to merge
    dfs_to_merge = [df_matches_lei_parent, df_matches_lei_owner, df_matches_lei_operator,
                    df_matches_permid_parent, df_matches_permid_owner, df_matches_permid_operator]

    # Initialize df_sfi_matches with the 'uid' column
    df_matches = df_asset_data[["uid"]]

    # Iteratively merge each dataframe in the list
    for df in dfs_to_merge:
        df_matches = pd.merge(df_matches, df, on="uid", how="left")

    # collect all matches in one column
    df_matches["matched_permid_identifier"] = df_matches["matched_permid_parent_permid"].combine_first(
        df_matches["matched_permid_parent_lei"]) \
        .combine_first(df_matches["matched_permid_owner_permid"]).combine_first(df_matches["matched_permid_owner_lei"]) \
        .combine_first(df_matches["matched_permid_operator_permid"]).combine_first(
        df_matches["matched_permid_operator_lei"])

    # delete irrelevant columns, i.e. the intermediate matches
    columns_to_drop = ["matched_permid_parent_permid", "matched_permid_parent_lei", "matched_permid_owner_permid",
                       "matched_permid_owner_lei", "matched_permid_operator_permid", "matched_permid_operator_lei"]
    df_matches = df_matches.drop(columns=columns_to_drop)

    # add df_matches to df_asset_data
    df_asset_data = pd.merge(df_asset_data, df_matches, on="uid", how="left")

    #### 1.2) MAPPING VIA DIRECT TEXT MATCHING
    print("Mapping via direct text matching...")

    # Use the function to add a new column 
    df_asset_data['direct_permid_parent_name'] = df_asset_data['parent_name'].apply(
        lambda company_name: get_permid_for_company(company_name, df_portfolio_name2permid))
    df_asset_data['direct_permid_owner_name'] = df_asset_data['owner_name'].apply(
        lambda company_name: get_permid_for_company(company_name, df_portfolio_name2permid))
    df_asset_data['direct_permid_operator_name'] = df_asset_data['operator_name'].apply(
        lambda company_name: get_permid_for_company(company_name, df_portfolio_name2permid))

    # combine results in one column
    df_asset_data.reset_index(drop=True, inplace=True)  # drop index due to duplicates
    df_asset_data['matched_permid_direct_text'] = df_asset_data['direct_permid_parent_name'].combine_first(
        df_asset_data['direct_permid_owner_name']).combine_first(df_asset_data['direct_permid_operator_name'])

    # delete irrelevant columns
    columns_to_drop = ["direct_permid_parent_name", "direct_permid_owner_name", "direct_permid_operator_name"]
    df_asset_data = df_asset_data.drop(columns=columns_to_drop)

    #### 1.3) MAPPING VIA FUZZY STRING MATCHING
    print("Mapping via fuzzy string matching...")

    # extract unmatched entries for fuzzy string matching (= no permid in matched_permid_direct_text & matched_permid_identifier)
    df_asset_unmatched = df_asset_data[
        df_asset_data['matched_permid_direct_text'].isnull() & df_asset_data['matched_permid_identifier'].isnull()]

    # apply map_fuzzy_matches function on standardized_ownership_parent_name & standardized_ownership_owner_name & standardized_ownership_operator_name
    score_cutoff = 95
    df_asset_data = map_fuzzy_matches("parent_name", df_asset_unmatched, df_asset_data, df_portfolio, score_cutoff)
    df_asset_data = map_fuzzy_matches("owner_name", df_asset_unmatched, df_asset_data, df_portfolio, score_cutoff)
    df_asset_data = map_fuzzy_matches("operator_name", df_asset_unmatched, df_asset_data, df_portfolio, score_cutoff)

    # combine fuzzy matches & clean up intermediate columns
    df_asset_data['matched_permid_fuzzy'] = df_asset_data['matched_permid_fuzzy_parent_name'].combine_first(
        df_asset_data['matched_permid_fuzzy_owner_name']).combine_first(
        df_asset_data['matched_permid_fuzzy_operator_name'])
    df_asset_data.drop(
        ['matched_permid_fuzzy_parent_name', 'matched_permid_fuzzy_owner_name', 'matched_permid_fuzzy_operator_name'],
        axis=1, inplace=True)

    ### 1.4) COMBINE ALL MATCHES
    df_asset_data = merge_and_count_matches(df_asset_data, ["matched_permid_identifier", "matched_permid_direct_text",
                                                            "matched_permid_fuzzy"])

    # -----------------------------------------------
    # 2) COMBINE, ANALYSE AND CLEAN
    # -----------------------------------------------

    '''
    To make the exercise above useful, we need to derive a joint dataset at permid-location level with the following columns:
        - permid (the permid at the asset level)
        - coordinate
        - coordinate 2 country (extract country from coordinate) -- to be done!
        - sector (NACE code!)
        - size indicator!
    '''

    df_asset_level = df_asset_data.copy()  # copy for further processing

    # define set of relevant columns & delete missing permid
    relevant_columns = ["uid", "matched_permid",
                        "latitude", "longitude",
                        "country", "sector",
                        "capacity", "capacity_unit", "data_source"]

    df_asset_level = df_asset_level[relevant_columns]
    df_asset_level = df_asset_level[df_asset_level["matched_permid"].notnull()]  # delete rows with missing permid

    ### 2.1) REMOVE DUPLICATES
    print("Removing duplicates...")

    # Assign numerical rankings to data sources
    data_source_rankings = {
        'SFI': 3,
        'GEM': 2,
        'Climate Trace': 1
    }

    # Haversine formula to calculate distance between two lat-lon points

    def haversine(lon1, lat1, lon2, lat2):
        """
        Calculate the great-circle distance between two points on the Earth surface.

        Uses the Haversine formula to calculate the distance between point A and B, provided their latitude and longitude.

        Parameters:
        - lon1: Longitude of point A in decimal degrees.
        - lat1: Latitude of point A in decimal degrees.
        - lon2: Longitude of point B in decimal degrees.
        - lat2: Latitude of point B in decimal degrees.

        Returns:
        - distance: Distance between point A and B in kilometers.
        """
        # Radius of the Earth in kilometers
        R = 6371.0
        # Convert latitude and longitude from degrees to radians
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        # Difference in coordinates
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        # Haversine formula
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = R * c
        return distance

    # Function to remove less favorable duplicates

    def remove_less_favorable_duplicates(df, threshold=5.0):  # Threshold in kilometers
        """
        Remove less favorable duplicate assets from a dataframe based on a proximity threshold and data source rankings.

        This function first sorts the df by the data_source_rankings. 
        For each asset, it checks for duplicates within a specified threshold distance. 
        If duplicates are found, the one with the lower data source rank (or later in the sorted list) is marked for removal.

        Parameters:
        - df: Pandas DataFrame containing asset-level data including 'matched_permid', 'latitude', 'longitude', and 'data_source'.
        - threshold: float, optional (default=0.1). The threshold distance in kilometers for considering two assets as duplicates.

        Returns:
        - df_deduped: Pandas DataFrame with less favorable duplicates removed.

        Example:
        - df_deduped = remove_less_favorable_duplicates(df_asset_level, threshold=0.1)
        """
        original_row_count = len(df)  # Store the original number of rows for quality control

        # Sort df by data_source rank (descending) so the highest rank comes first
        df['data_source_rank'] = df['data_source'].map(data_source_rankings)
        df_sorted = df.sort_values(by=['data_source_rank'], ascending=False)

        to_drop = []  # Placeholder for indices to drop

        # Iterate over each unique 'matched_permid'
        for _, group in df_sorted.groupby('matched_permid'):
            # Further iterate over each pair within the group
            for i, row in group.iterrows():
                for j, compare_row in group.iterrows():
                    if i >= j:  # Prevent comparing the row with itself and ensure each pair is only evaluated once
                        continue
                    if row['data_source'] == compare_row['data_source']:
                        continue  # Skip if from the same data source (here we trust that assets are distinct)
                    # Check for identical latitude or longitude
                    if row['latitude'] == compare_row['latitude'] or row['longitude'] == compare_row['longitude']:
                        to_drop.append(j)  # if identical, add lower-ranked source to the list
                    else:
                        # Calculate distance for entries not exactly matching in lat/long
                        distance = haversine(row['longitude'], row['latitude'], compare_row['longitude'],
                                             compare_row['latitude'])
                        if distance <= threshold:
                            to_drop.append(j)  # Mark the lower-ranked entry for removal

        # Drop less favorable duplicates based on indices collected
        df_deduped = df_sorted.drop(index=to_drop).drop(columns=['data_source_rank'])

        # Calculate and print the number of rows removed
        rows_removed = original_row_count - len(df_deduped)
        print(f"Removed {rows_removed} rows out of {original_row_count} total rows.")

        return df_deduped

    # Apply the function (create asset_level_reduced)
    df_asset_level_red = remove_less_favorable_duplicates(df_asset_level,
                                                          threshold=5)  # Threshold in kilometers

    ### 2.2) SECTOR - NACE Mapping
    print("Assigning NACE sector codes. This needs to be updated if the input data changes.")

    # create dictionary to map sector information (exracted from SFI, GEM, CLT) to NACE
    sector_to_nace = {
        "beef/slaughter": "10.11",
        "cement/dry": "23.51",
        "cement/": "23.51",
        "steel/coke": "24.10",
        "steel/crude steel": "24.10",
        "steel/": "24.10",
        "steel/iron": "24.10",
        "steel/finished products": "24.20",
        "steel/pellets": "24.10",
        "pulp paper/paper": "17.11",
        "pulp paper/pulp and paper": "17.11",
        "pulp paper/pulp": "17.11",
        "petrochemicals/Ammonia": "20.11",
        "petrochemicals/Ethylene": "20.13",
        "petrochemicals/Propylene": "20.13",
        "petrochemicals/Butadiene": "20.13",
        "petrochemicals/Benzene": "20.13",
        "petrochemicals/Toulene/Xylene": "20.13",
        "petrochemicals/Methanol": "20.13",
        "pulp paper/recycled paper": "17.12",
        "pulp paper/pulp ": "17.11",
        "wastewater/yes": "37.00",
        "cement/wet": "23.51",
        "coal plant/unknown": "35.11",
        "coal plant/bituminous": "35.11",
        "coal plant/subbituminous": "35.11",
        "coal plant/lignite": "35.11",
        "coal plant/anthracite": "35.11",
        "coal plant/waste coal": "35.11",
        "LNG terminal/import": "49.50",
        "coal mine/underground": "05.10",
        "coal mine/surface": "05.20",
        "coal mine/underground & surface": "05.10",
        "wind power/onshore": "35.11",
        "bioenergy": "35.11",
        "LNG terminal/export": "49.50",
        "solar power/pv": "35.11",
        "solar power/assumed pv": "35.11",
        "solar power/solar thermal": "35.11",
        "steel": "24.10",
        "wind power/offshore hard mount": "35.11",
        "wind power/offshore floating": "35.11",
        "oil & gas extraction/oil and gas": "06.20",
        "nuclear/pressurized water reactor": "35.11",
        "nuclear/boiling water reactor": "35.11",
        "oil & gas extraction/gas": "06.20",
        "hydropower/pumped storage": "35.11",
        "hydropower/conventional storage": "35.11",
        "hydropower/conventional and run-of-river": "35.11",
        "hydropower/run-of-river": "35.11",
        "hydropower/unknown": "35.11",
        "coal terminal/coal": "52.22",
        "coal terminal/all cargo": "52.22",
        "geothermal/flash steam - double": "35.11",
        "geothermal/binary cycle": "35.11",
        "geothermal/flash steam - triple": "35.11",
        "oil & gas extraction/oil": "06.20",
        "iron mining": "07.10",
        "international aviation": "51.10",
        "electricity generation": "35.11",
        "oil and gas refining": "19.20",
        "oil and gas production and transport": "06.10",
        "pulp and paper": "17.11",
        "copper mining": "07.29",
        "coal mining": "05.10",
        "domestic aviation": "51.10",
        "chemicals": "20.59",
        "cement": "23.51",
        "bauxite mining": "07.29",
        "aluminum": "24.10"
    }

    # create new column based on dictionary
    df_asset_level_red["nace_code"] = df_asset_level_red["sector"].map(sector_to_nace)
    # NOTE: NACE code 35.11 is very broad and includes a variety of energy generation methods. 

    ### 2.3) WEIGHTING

    ''' NOTES
    To deal with the problem of missing capacity information, we implement the following logic:
    - A) Clean capacity column (replace non-numeric values with NaN)
    - B) Impute missing information if for the specific sector (= capacity unit!) the permid has more than 50% capacity information
    - C) Weighting across sectors:
        Due to the merge of different asset-level data sources, some companies have more than one sector and different capacity information.
        Hence, we determine the overall importance of a sector for a company based on the relative number of assets.
        Example: If a company has 50 assets, 40 in beef, 10 in biofuel, we derive weights based on the number of assets (i.e., 0.8 for beef, 0.2 for biofuel).
    '''

    ### A: Clean capacity column (replace non-numeric values with NaN)
    df_asset_level_red["capacity"] = pd.to_numeric(df_asset_level_red["capacity"], errors='coerce')
    df_asset_level_red["capacity_unit"].replace("mt per year", "mtpa",
                                                inplace=True)  # replace "mt per year" with "mtpa", the rest is already streamlined

    ### B: Impute missing information if for the specific sector the permid has more than 50% capacity information

    # create new "category" in capacity_unit column: replace missing values with "unknown"
    # Logic: also for this "category", we calculate the share of assets per permid
    df_asset_level_red["capacity_unit"].fillna("unknown", inplace=True)

    # Function to apply imputation logic to each group
    def impute_group(group):
        """
        Imputes missing values in the 'capacity' and 'capacity_unit' for a given group.
        Adds a binary column 'capacity_modified' to indicate rows where 'capacity' was imputed.
        Logic: If more than 50% of rows per company (permid) and sector(=capacity unit!) have capacity information, 
            impute missing values for 'capacity' with the median of the group and 'capacity_unit' with the mode of the group.

        Parameters:
        - group (DataFrame): The group to impute missing values for.

        Returns:
        - DataFrame: The group with imputed missing values and an indicator for 'capacity' imputation.
        """
        capacity_notna_ratio = group['capacity'].notna().mean()
        group['capacity_modified'] = 0  # Initialize the indicator column with 0s

        if capacity_notna_ratio > 0.5:
            # Check for rows where 'capacity' is NaN to mark them before imputation
            capacity_missing_indices = group[group['capacity'].isna()].index
            # Impute 'capacity' with median of the group
            group.loc[capacity_missing_indices, 'capacity'] = group['capacity'].median()
            group.loc[capacity_missing_indices, 'capacity_modified'] = 1  # Mark as modified

            # Impute 'capacity_unit' with the mode of the group if needed
            if group['capacity_unit'].isna().any():
                most_frequent_unit = group['capacity_unit'].mode()[0]
                group['capacity_unit'].fillna(most_frequent_unit, inplace=True)

        return group

    # Apply the imputation function to each 'matched_permid' and 'sector'(=capacity unit!) group
    df_asset_level_red = df_asset_level_red.groupby(['matched_permid', 'capacity_unit']).apply(impute_group)
    df_asset_level_red = df_asset_level_red.reset_index(drop=True)

    # How many rows have been modified?
    print(f"Modified {df_asset_level_red['capacity_modified'].sum()} rows out of {len(df_asset_level_red)} total rows.")

    ### C: Weighting across sectors

    # C1: Calculate sector weights based on asset counts across different sectors/capacity units
    ''' 
    NOTE that we take the capacity_unit column here! 
    This differs quite a lot compared to "sector". 
    In the sector column we tried to collect as much information as possible from the different data sources.
    Example: solar power/pv , or solar power/assumed pv, or solar power/solar thermal, etc.

    In the capacity_unit column we have a more standardized view on the capacity units. (e.g, MW for energy generation)
    '''

    # Calculate the weights for each sector based on the number of assets
    sector_weights = df_asset_level_red.groupby(['matched_permid', 'capacity_unit']).size().div(
        df_asset_level_red.groupby('matched_permid').size(), level='matched_permid'
    ).reset_index(name='weight')

    # Merge sector weights back to the main DataFrame
    df_asset_level_red = df_asset_level_red.merge(sector_weights, on=['matched_permid', 'capacity_unit'], how='left')

    # C2: Adjust weights within sectors based on capacity
    def adjust_weights_within_sectors(group):
        """
        Adjusts the weights within a group based on the capacity of each item.

        If any capacity is missing within the group, the weight is distributed equally among the items.
        Otherwise, the weight is distributed based on the proportional capacity of each item.

        Parameters:
        - group: pandas.DataFrame (here: the group with the same 'matched_permid' and 'capacity_unit')
        
        Returns:
        - pandas.DataFrame: The group with adjusted weights based on capacity.
        """
        if group['capacity'].isnull().any():
            # If any capacity is missing within the group, distribute weight equally
            group['final_weight'] = group['weight'] / len(group)
        else:
            # Distribute weight based on proportional capacity
            total_capacity = group['capacity'].sum()
            group['final_weight'] = (group['capacity'] / total_capacity) * group['weight']
        return group

    # Apply weight adjustments within each sector
    df_asset_level_red = df_asset_level_red.groupby(['matched_permid', 'capacity_unit']).apply(
        adjust_weights_within_sectors)
    df_asset_level_red = df_asset_level_red.reset_index(drop=True)
    print('Weights adjusted.')

    ### 3.4) Coordinate to country
    ''' 
    For IO modelling, for example, we need to know the country of the asset.
    We can extract this information from A) the unstructured "country" column, or B) via the coordinates.
    '''

    # Preliminary mapping for common countries to their ISO codes
    preliminary_mapping = {
        "China": "CN",
        "United States": "US",
        "India": "IN",
        "Brazil": "BR",
        "Congo, Democratic Republic of the": "CD",
        "Turkey": "TR",
        "Russia": "RU",
        "Türkiye": "TR"
    }

    # Function to standardize country to ISO codes with preliminary mapping and direct ISO code check
    def standardize_country(country):
        """
        Standardize country names or codes to ISO 3166-1 alpha-2 codes.

        This function attempts to standardize a given country name or code to its corresponding ISO 3166-1 alpha-2 code.
        It first checks if the country is in a preliminary mapping dictionary. If not, it checks if the input is a valid
        2-letter ISO code. For other cases, it attempts to find the country using the `pycountry` library.

        Args:
            country (str): The country name or code to be standardized.

        Returns:
            str or None: The standardized ISO 3166-1 alpha-2 code if found; otherwise, None.
        """
        if country in preliminary_mapping:
            return preliminary_mapping[country]
        elif len(country) == 2 and pycountry.countries.get(alpha_2=country):
            return country  # Assume 3-letter entries are valid ISO codes and exist in pycountry
        else:
            # Attempt to find the country in pycountry for other cases
            try:
                return pycountry.countries.lookup(country).alpha_2
            except LookupError:
                return None  # Return None or a placeholder if no ISO code found

    # Apply the standardization function to the 'country' column
    df_asset_level_red['country_iso'] = df_asset_level_red['country'].apply(standardize_country)

    # Quality check: Identify rows with None as 'country_iso' indicating unsuccessful standardization
    unstandardized_rows = df_asset_level_red[df_asset_level_red['country_iso'].isnull()]
    if len(unstandardized_rows) > 0:
        print(f"Rows needing further review or manual correction: {len(unstandardized_rows)}")
        print(unstandardized_rows[
                  ['country', 'country_iso']].head())  # Preview rows that may require manual review or correction
    else:
        print("Everything seems fine.")

    # -----------------------------------------------
    # 3) EXPORT
    # -----------------------------------------------
    df_asset_level_red.rename(columns={"matched_permid": "permid"}, inplace=True)  # rename matched_permid to permid

    # keep only essential columns
    essential_columns_aggregated = ["permid", "country_iso", "nace_code", "final_weight"]
    essential_columns_disaggregated = ["permid", "longitude", "latitude", "country_iso", "nace_code", "final_weight"]
    df_asset_level_aggregated = df_asset_level_red[essential_columns_aggregated]
    df_asset_level_disaggregated = df_asset_level_red[essential_columns_disaggregated]

    # Aggregate to get unique permid-region-sector pairs with their final_weights
    df_asset_level_aggregated = df_asset_level_aggregated.groupby(['permid', 'nace_code', 'country_iso'])[
        'final_weight'].sum().reset_index()

    # EXPORT
    df_asset_level_disaggregated.to_csv(
        Path(PATH_TO_OUTPUT_FOLDER) / "internal_data/asset_level_data_disaggregated.csv", index=False)
    df_asset_level_aggregated.to_csv(Path(PATH_TO_OUTPUT_FOLDER) / "internal_data/asset_level_data_aggregated.csv",
                                     index=False)

    print("Asset-level data has been successfully merged with portfolio company data.")
    return df_asset_level_disaggregated, df_asset_level_aggregated
