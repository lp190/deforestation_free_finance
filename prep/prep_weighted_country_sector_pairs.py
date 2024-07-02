"""
Description:
    This script integrates asset-level data, ORBIS data, and disaggregated revenue data to compute
    weighted sector-region pairs for companies in a portfolio.

Output:
    The script returns a DataFrame with the following columns:
        - 'identifier'  : Basically any unique company identifier
        - 'country_iso' : 2-digit ISO country code
        - 'nace_code'   : NACE industry classification code
        - 'weight_final': Final weighted average based on specified importance set by the user

NOTES:
    - Ensure the importance values for each data source are correctly set.
    - This script requires the presence of several data files as specified in the function arguments.
"""

# import packages
import pandas as pd
import pycountry


#####    ####    ####    ####    ####    ####    ####    #####
####   Integration of Asset-Level and Headquarter DATA    ####
#####    ####    ####    ####    ####    ####    ####    #####

## STEP 1 - load all possibly deployed dfs
def prep_weighted_country_sector_pairs(asset_data, portfolio_data, bias_towards_existing_data,
                                       bias_towards_missing_data, importance_asset_info, importance_headquarter_info):
    """
    This function takes in several dataframes and performs a series of operations to prepare a final dataframe.

    Args:
        asset_data (DataFrame): Contains at least the columns ['identifier', 'country_iso', 'nace_code', 'final_weight']
        portfolio_data (DataFrame): Contains at least the columns ['identifier', 'country_iso', 'nace_code']
        bias_towards_existing_data (bool): If True, the function processes the data per company, otherwise it averages the data
        bias_towards_missing_data (bool): If True, the function processes the data by averaging, otherwise it processes the data per company
        importance_asset_info (float): The importance value for asset information
        importance_headquarter_info (float): The importance value for headquarter information

    Returns:
        processed_dfs (DataFrame): The final dataframe with the columns
                                    ['identifier', 'country_iso', 'nace_code', 'weight_final']
    """

    # Step 1: Data Preparation
    # Selecting relevant columns from each dataframe

    # Asset-Level Data - Climate Trace, Spatial Finance Initiative,...
    asset_data = asset_data[['identifier', 'country_iso', 'nace_code', 'final_weight']]
    asset_data.columns = ['identifier', 'country_iso', 'nace_code', 'weight_asset']  # Rename columns

    # Manual cleaning: replace country code Jersey (JE) with UK (GB)
    asset_data['country_iso'] = asset_data['country_iso'].replace('JE', 'GB')

    # HQ Data 
    portfolio_data = portfolio_data[['identifier', 'country_iso', 'nace_code']]
    portfolio_data['weight_headquarter'] = 1  # As a company only has one HQ the weight is always 1

    ## STEP 2 - Aggregate all available dfs
    # Merge all dataframes
    merged_df = pd.merge(asset_data, portfolio_data, on=['identifier', 'country_iso', 'nace_code'], how='outer').fillna(
        0)

    ## STEP 3 - Combine final country-sector pairs from each data comilation per company (depending on the availability)

    # Check whether there are any invalid country ISO codes
    valid_country_iso_codes = {country.alpha_2 for country in pycountry.countries}
    invalid_codes = merged_df[~merged_df['country_iso'].isin(valid_country_iso_codes)]['country_iso']

    if not invalid_codes.empty:
        invalid_codes_list = invalid_codes.tolist()
        print(f"Invalid country ISO codes found: {invalid_codes_list}. These observations will be dropped.")

        # Drop the observations with invalid country ISO codes
        merged_df = merged_df[merged_df['country_iso'].isin(valid_country_iso_codes)]
    else:
        print("All country ISO codes are valid.")

    # Assume column_weights and importance values are defined outside this function
    # Define importance values for each weight column (example values)
    # Update column_weights to use these importance values
    column_weights_importance = {
        'weight_asset': importance_asset_info,
        'weight_headquarter': importance_headquarter_info
    }

    ## STEP 3.1 Averaging row by row
    # Function to calculate weighted average for a row
    def weighted_avg(row, column_weights):
        """
        Calculate the weighted average of a row based on the values and importance of columns.

        Args:
            row (pd.Series): A row from a DataFrame.
            column_weights (dict): A dictionary with column names as keys and importance values as values.

        Returns:
            float: The weighted average of the row. Returns 0 if the total importance is 0.
        """
        # Initialize variables to store the weighted sum and total importance
        weighted_sum = 0
        total_importance = 0

        # Iterate over each column and its importance value
        for col, importance in column_weights.items():
            # If the value in the row is greater than 0
            if row[col] > 0:
                # Add the product of the value and importance to the weighted sum
                weighted_sum += row[col] * importance
                # Add the importance to the total importance
                total_importance += importance

        # Calculate the weighted average by dividing the weighted sum by the total importance
        # If the total importance is 0, return 0
        return weighted_sum / total_importance if total_importance > 0 else 0

    def process_dataframe(df, column_weights_importance_unequal):
        """
        Process the input DataFrame by applying the weighted averaging function.
        Args:
            df (pd.DataFrame): The input DataFrame.
            column_weights_importance_unequal (dict): A dictionary containing column weights and their importance values.
        Returns:
            pd.DataFrame: The final DataFrame with 'isin', 'country_iso', 'nace_code', and 'weight_final' columns.
        """
        # Apply averaging function
        df['weight_final'] = df.apply(weighted_avg, args=(column_weights_importance_unequal,), axis=1)
        # Subset certain columns for the final DataFrame
        final_df = df[['identifier', 'country_iso', 'nace_code', 'weight_final']]
        return final_df

    ## STEP 3.2 Averaging the datasets based on available data per company
    # Adapted calculate_weight_final function with importance values
    def calculate_weight_final(row, importance_asset_info, importance_headquarter_info):
        """
        Calculates the weighted final value based on the input row and importance values.

        Parameters:
            row (DataFrame): The row containing the data for calculation.
            importance_asset_info (float): The importance value for asset information.
            importance_headquarter_info (float): The importance value for headquarter information.

        Returns:
            float: The calculated weighted final value.
        """
        if row['data_for_weight_asset_and_weight_headquarter_exists'] == 1:
            return (row['weight_asset'] * importance_asset_info +
                    row['weight_headquarter'] * importance_headquarter_info
                    ) / (importance_asset_info + importance_headquarter_info)
        elif row['data_for_weight_asset_exists'] == 1:
            return row['weight_asset']
        elif row['data_for_weight_headquarter_exists'] == 1:
            return row['weight_headquarter']
        else:
            return 0  # Default case if none of the binary conditions are met

    def process_dataframe_per_company(df):
        """
        Process the dataframe per company by creating binary columns based on non-zero weights and updating additional
        binary columns based on conditions.

        Parameters:
            df (DataFrame): Input dataframe containing the necessary columns for processing.

        Returns:
            DataFrame: Processed dataframe with added binary columns and updated binary columns.
        """
        # Creating binary columns based on non-zero weights

        for col in ['weight_asset', 'weight_headquarter']:
            df[f'data_for_{col}_exists'] = df.groupby('identifier')[col].transform(lambda x: (x != 0).any().astype(int))

        # Initializing additional binary columns to 0
        df['data_for_weight_asset_and_weight_headquarter_exists'] = 0

        # Updating additional binary columns based on conditions
        df.loc[(df['data_for_weight_asset_exists'] == 1) & (df['data_for_weight_headquarter_exists'] == 1),
               'data_for_weight_asset_and_weight_headquarter_exists'] = 1

        # Apply calculate_weight_final function
        df['weight_final'] = df.apply(calculate_weight_final,
                                      args=(importance_asset_info, importance_headquarter_info),
                                      axis=1)
        return df

    ## STEP 3.1:
    if bias_towards_missing_data:
        processed_dfs = process_dataframe(merged_df, column_weights_importance)
    ## STEP 3.2:
    elif bias_towards_existing_data:
        processed_dfs = process_dataframe_per_company(merged_df)

    return processed_dfs
