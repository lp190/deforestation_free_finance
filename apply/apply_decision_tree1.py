"""
Description:
    This script applies decision tree 1:
        - Function 1: conservative approach (buckets)
        - Function 2: weighted average approach (score)
"""

import pandas as pd
from scipy.stats.mstats import winsorize

from utils import dt1_scoring_calculate_weighted_score, dt1_scoring_log_transform_and_normalize


def apply_dt1_conservative_approach(df_portfolio,
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
                                    hotspot_subsidiaries_threshold):
    """
    Apply a conservative decision tree model to filter and categorize data based on the different variables + cutoffs.

    Args:
        df_portfolio (DataFrame): DataFrame containing the data to be analyzed.
        use_io_model_score (bool): Whether to use IO model scores for filtering.
        use_trase_flag (bool): Whether to include the 'trase_flag'.
        use_flag_direct (bool): Whether to include the 'final_flag_direct_score'.
        use_flag_indirect (bool): Whether to include the 'final_flag_indirect_score'.
        use_flag_forest500 (bool): Whether to include the 'flag_forest500'.
        recent_controversies_cutoffs (bool): Whether to apply recent controversies cutoffs.
        historical_controversies_cutoffs (bool): Whether to apply historical controversies cutoffs.
        flag_direct_threshold_high (float): High threshold for 'final_flag_direct_score'.
        flag_direct_threshold_medium (float): Medium threshold for 'final_flag_direct_score'.
        flag_indirect_threshold_high (float): High threshold for 'final_flag_indirect_score'.
        flag_indirect_threshold_medium (float): Medium threshold for 'final_flag_indirect_score'.
        IO_threshold_high (float): High threshold for IO model scores.
        IO_threshold_medium (float): Medium threshold for IO model scores.
        recent_controversies_threshold_high (float): High threshold for recent controversies.
        recent_controversies_threshold_medium (float): Medium threshold for recent controversies.
        historical_controversies_threshold_high (float): High threshold for historical controversies.
        historical_controversies_threshold_medium (float): Medium threshold for historical controversies.
        cutoff_direct_attribution (float): Cutoff for direct attribution scores.
        subsidiary_data_exists (bool): Indicates whether subsidiary data is available for hotspot analysis.
        hotspot_assets_threshold (float): Threshold value for identifying hotspot assets.
        hotspot_subsidiaries_threshold (float): Threshold value for identifying hotspot subsidiaries.

    Returns:
        DataFrame: DataFrame containing the original portfolio data with an additional dt1_conservative column.
    """

    dfs_high = []
    dfs_medium = []

    ### SECTORAL ###

    # flags direct
    if use_flag_direct:
        flag_direct_high = df_portfolio[df_portfolio['flag_direct_score'] > flag_direct_threshold_high]
        dfs_high.append(flag_direct_high)

        flag_direct_medium = df_portfolio[(df_portfolio['flag_direct_score'] > flag_direct_threshold_medium) &
                                          (df_portfolio['flag_direct_score'] <= flag_direct_threshold_high)]
        dfs_medium.append(flag_direct_medium)

    # flagged by TRASE
    if use_trase_flag:
        flag_trase_positive = df_portfolio[df_portfolio['trase_flag'] == 1]
        dfs_high.append(flag_trase_positive)
        # will be removed later again
        dfs_medium.append(flag_trase_positive)

    # DIRECT ATTRIBUTION
    # if cutoff is zero, do not include the zero
    if cutoff_direct_attribution == 0.0:
        direct_attribution = df_portfolio[df_portfolio['direct_attribution_score'] > cutoff_direct_attribution]
    else:
        direct_attribution = df_portfolio[df_portfolio['direct_attribution_score'] >= cutoff_direct_attribution]
    dfs_high.append(direct_attribution)
    # will be removed later again
    dfs_medium.append(direct_attribution)

    # INDIRECT EXPOSURE

    # Filter rows that are above the cutoff
    if use_io_model_score:
        # assign to high if above threshold
        indirect_IO_high = df_portfolio[df_portfolio['io_supply_chain_score'] >= IO_threshold_high]
        dfs_high.append(indirect_IO_high)

        # assign to medium if between thresholds
        indirect_IO_medium = df_portfolio[(df_portfolio['io_supply_chain_score'] >= IO_threshold_medium) &
                                          (df_portfolio['io_supply_chain_score'] < IO_threshold_high)]
        dfs_medium.append(indirect_IO_medium)

    # flags indirect:
    if use_flag_indirect:
        flag_indirect_high = df_portfolio[df_portfolio['flag_indirect_score'] > flag_indirect_threshold_high]
        dfs_high.append(flag_indirect_high)
        flag_indirect_medium = df_portfolio[(df_portfolio['flag_indirect_score'] > flag_indirect_threshold_medium) &
                                            (df_portfolio['flag_indirect_score'] <= flag_indirect_threshold_high)]
        dfs_medium.append(flag_indirect_medium)

    ### CONTROVERSIES ###

    if use_flag_forest500:
        flag_forest500 = df_portfolio[df_portfolio['flag_forest500'] == 1.0]
        dfs_high.append(flag_forest500)
        # will be removed later again
        dfs_medium.append(flag_forest500)

    if recent_controversies_cutoffs:
        # Note only ~100 companies have a recent controversy. So percentile cutoffs are not very useful.
        recent_controversies_high = df_portfolio[df_portfolio['var_impact_controv_recent'] >=
                                                 recent_controversies_threshold_high]
        dfs_high.append(recent_controversies_high)
        recent_controversies_medium = df_portfolio[df_portfolio['var_impact_controv_recent'] ==
                                                   recent_controversies_threshold_medium]
        dfs_medium.append(recent_controversies_medium)

    if historical_controversies_cutoffs:
        # if cutoff is zero, do not include the zero
        if historical_controversies_threshold_high == 0.0:
            historical_controversies_high = df_portfolio[df_portfolio['var_impact_controv_count'] >
                                                         historical_controversies_threshold_high]
        else:
            historical_controversies_high = df_portfolio[df_portfolio['var_impact_controv_count'] >=
                                                         historical_controversies_threshold_high]

        dfs_high.append(historical_controversies_high)
        # repeat for medium bucket
        if historical_controversies_threshold_medium == 0.0:
            historical_controversies_medium = df_portfolio[df_portfolio['var_impact_controv_count'] >
                                                           historical_controversies_threshold_medium]
        else:
            historical_controversies_medium = df_portfolio[df_portfolio['var_impact_controv_count'] >=
                                                           historical_controversies_threshold_medium]

        dfs_medium.append(historical_controversies_medium)

    if subsidiary_data_exists:
        hotspot_high = df_portfolio[df_portfolio['asset_impact_assignment_count_subsidiary'] >=
                                    hotspot_subsidiaries_threshold | df_portfolio[
                                        'asset_impact_assignment_count_asset'] >= hotspot_assets_threshold]

    else:
        hotspot_high = df_portfolio[df_portfolio['asset_impact_assignment_count_asset'] >= hotspot_assets_threshold]

    # Combine different dataframes
    high_exposure_total = pd.concat(dfs_high).drop_duplicates().reset_index(drop=True)
    medium_and_high = pd.concat(dfs_medium).drop_duplicates().reset_index(drop=True)

    # Determine high & medium buckets via merging
    merged_df = pd.merge(medium_and_high, high_exposure_total, how='outer', indicator=True)
    medium_exposure_total = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])

    # we only change the bucketing between high and very high; let's split between high and very high here
    length_high_before_splitting = len(high_exposure_total)

    # Perform an inner join to get only entries with isin in both DataFrames
    very_high_exposure_total = pd.merge(high_exposure_total, hotspot_high, on='identifier', how='inner', indicator=True)
    high_exposure_total = high_exposure_total[~high_exposure_total['identifier'].isin(very_high_exposure_total['identifier'])]
    very_high_exposure_total.drop(columns=['_merge'], inplace=True)

    # now check that the length of very high and high equals the high of before

    if not length_high_before_splitting == (len(very_high_exposure_total) + len(high_exposure_total)):
        raise ValueError('something is wrong in the splitting of the high and very high exposure dataframes')

    # remove the merge column; so we can do merging again to define the "low" bucket
    merged_df = merged_df.drop(columns=['_merge'])
    merged_df_low = pd.merge(df_portfolio, merged_df, how='left', indicator=True)
    low_exposure_total = merged_df_low[merged_df_low['_merge'] == 'left_only'].drop(columns=['_merge'])

    if not len(low_exposure_total) + len(medium_exposure_total) + len(high_exposure_total) + len(very_high_exposure_total) == len(
            df_portfolio):
        print(len(low_exposure_total))
        print(len(medium_exposure_total))
        print(len(high_exposure_total))
        print(len(very_high_exposure_total))
        print(len(low_exposure_total) + len(medium_exposure_total) + len(high_exposure_total) + len(very_high_exposure_total))
        print(len(df_portfolio))
        raise ValueError('something is wrong: buckets do not add up')

    # Create the dt1_conservative column
    df_portfolio['dt1_conservative'] = 'low'
    df_portfolio.loc[df_portfolio['identifier'].isin(medium_exposure_total['identifier']), 'dt1_conservative'] = 'medium'
    df_portfolio.loc[df_portfolio['identifier'].isin(high_exposure_total['identifier']), 'dt1_conservative'] = 'high'
    df_portfolio.loc[
        df_portfolio['identifier'].isin(very_high_exposure_total['identifier']), 'dt1_conservative'] = 'very high'

    # Print statement for quality control
    print('Column added to the DataFrame: dt1_conservative')
    print('Distribution of dt1_conservative:')
    print(df_portfolio['dt1_conservative'].value_counts())

    return df_portfolio


def apply_dt1_weighted_average_approach(df_portfolio,
                                        weight_dictionary_companies,
                                        weight_dictionary_financial_institutions,
                                        cutoffs=(0.9, 0.7)):
    """
    Apply a weighted average approach to assess exposure levels of companies and financial institutions in a portfolio.

    NB. one could consider to make the cutoffs changeable per variable.

    Args:
        df_portfolio (DataFrame): DataFrame containing the portfolio data.
        weight_dictionary_companies (dict): Dictionary of weights for company-related scores.
        weight_dictionary_financial_institutions (dict): Dictionary of weights for financial institution-related scores.
        cutoffs (tuple): Quantile cutoffs for determining exposure levels (default is (0.9, 0.7)).

    Returns:
        DataFrame: DataFrame containing the original portfolio data with additional dt1_score, dt1_score_winsorized,
                   and dt1_bucket_based_on_score columns.
    """

    # Store initial columns for print statement
    initial_columns = df_portfolio.columns

    # Because some columns are not yet between 0 and 1 we apply a log transformation followed by a normalization
    columns_to_normalize = list(
        set(weight_dictionary_companies.keys()).union(weight_dictionary_financial_institutions.keys()))

    # Apply transformation and normalization, replacing zeros
    for column in columns_to_normalize:
        df_portfolio[column] = dt1_scoring_log_transform_and_normalize(df_portfolio[column])

    # Apply weighting and calculate dt1_score (for companies in the portfolio)
    nace_of_financial_institutions = ['64', '65', '66']
    df_portfolio_companies = df_portfolio[
        ~df_portfolio['nace_code'].fillna('').str.startswith(tuple(nace_of_financial_institutions))]
    df_portfolio_financial_institutions = df_portfolio[
        df_portfolio['nace_code'].fillna('').str.startswith(tuple(nace_of_financial_institutions))]

    # Apply the weighting to Company-DF and Financial Institution-DF
    df_portfolio_companies['dt1_score'] = df_portfolio_companies.apply(
        lambda row: dt1_scoring_calculate_weighted_score(row, weight_dictionary_companies), axis=1)

    df_portfolio_financial_institutions['dt1_score'] = df_portfolio_financial_institutions.apply(
        lambda row: dt1_scoring_calculate_weighted_score(row, weight_dictionary_financial_institutions), axis=1)

    # Concatenate the two dataframes
    index_weight_filtered = pd.concat([df_portfolio_companies, df_portfolio_financial_institutions])

    # Normalize the dt1_score but also create another dt1_score which is winsorized before
    index_weight_filtered['dt1_score_winsorized'] = winsorize(index_weight_filtered['dt1_score'], limits=(0, 0.01))
    index_weight_filtered['dt1_score'] = dt1_scoring_log_transform_and_normalize(index_weight_filtered['dt1_score'])
    index_weight_filtered['dt1_score_winsorized'] = dt1_scoring_log_transform_and_normalize(
        index_weight_filtered['dt1_score_winsorized'])

    # Rank based on the score
    index_weight_filtered = index_weight_filtered.sort_values(by='dt1_score', ascending=False)

    # Assign exposure buckets based on the quantiles
    cutoff_high = index_weight_filtered['dt1_score'].quantile(cutoffs[0])
    cutoff_medium = index_weight_filtered['dt1_score'].quantile(cutoffs[1])

    index_weight_filtered['dt1_bucket_based_on_score'] = 'low'
    index_weight_filtered.loc[
        index_weight_filtered['dt1_score'] >= cutoff_medium, 'dt1_bucket_based_on_score'] = 'medium'
    index_weight_filtered.loc[index_weight_filtered['dt1_score'] >= cutoff_high, 'dt1_bucket_based_on_score'] = 'high'

    # Merge the dt1_score, dt1_score_winsorized, and dt1_bucket_based_on_score back to df_portfolio
    df_portfolio = df_portfolio.merge(
        index_weight_filtered[['dt1_score', 'dt1_score_winsorized', 'dt1_bucket_based_on_score']],
        left_index=True, right_index=True, how='left')

    # Print columns that were added to the DataFrame
    added_columns = list(set(df_portfolio.columns) - set(initial_columns))
    print(f"Columns added to the DataFrame: {added_columns}")

    return df_portfolio
