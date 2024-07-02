"""
Filename: apply_forest_and_finance_filter.py

Description:
    This script integrates forest and finance data into the input dataframe.
    The updated index weights dataframe with added forest and finance related information is returned.

Notes:
    - False positive list has been constructed manually. User would potentially have to adapt and could improve the
        fuzzy string matching
"""

from pathlib import Path

import pandas as pd

from filepaths import PATH_TO_OUTPUT_FOLDER, PATH_TO_FOREST_FINANCE
from generate.generate_forest_finance_scores import generate_forest_finance_scores
from user_input import FOREST_FINANCE_YOI, FOREST_FINANCE_TRUNCATION, FOREST_FINANCE_THRESHOLD
from utils import clean_company_name_new, find_best_match


def apply_forest_and_finance(df_portfolio, sector_column, financial_sector_codes):
    """
    Adds the forest and finance data to the portfolio index weights dataframe: note these are always zero for all
    non-FI companies.

    NOTE: handling false positives in matching the names of the forest finance dataset with the names of portfolio
            companies is a manual process and would need to be done per portfolio by the user.

    Args:
        df_portfolio (pd.DataFrame): dataframe containing at least the column specified by `sector_column`
        sector_column (str): The name of the column containing sector codes.
        financial_sector_codes (list): List of sector codes that are considered financial.


    Returns:
        df_portfolio (pd.DataFrame): the input dataframe, but now with the forest and finance related information:
                                    "forest_and_finance_amount" - SUM! of the forest and finance loans/equity/etc over
                                                                    the relevant timeperiod
                                    "forest_and_finance_score" - forest_and_finance_amount but then min-max scaled
                                    "forest_and_finance_flag" - binary variable that checks if forest_and_finance_amount
                                                                is above or below the FOREST_FINANCE_THRESHOLD
    """

    forest_finance_path = Path(PATH_TO_OUTPUT_FOLDER) / 'internal_data/forest_finance.csv'

    if forest_finance_path.exists():
        print('loading: Forest & Finance data...')
        forest_and_finance_scores = pd.read_csv(forest_finance_path)
        print('DONE')
    else:
        print('generating: forest_finance.csv')
        # Note the sum_on_cleaned_name=True setting, which aggregates the Forest & Finance on the cleaned company name.
        # This could be improved via a manual mapping.
        forest_and_finance_scores = generate_forest_finance_scores(PATH_TO_FOREST_FINANCE, FOREST_FINANCE_YOI,
                                                                   FOREST_FINANCE_TRUNCATION,
                                                                   sum_on_cleaned_name=True)
        forest_and_finance_scores.to_csv(forest_finance_path, index=False)
        print('DONE')

    # Store initial columns for print statement
    initial_columns = set(df_portfolio.columns)

    # Now we select the companies in the portfolio with a sector code equal to financials
    df_portfolio_financials = df_portfolio[df_portfolio[sector_column].isin(financial_sector_codes)]
    df_portfolio_nonfinancials = df_portfolio[~df_portfolio[sector_column].isin(financial_sector_codes)]

    # Now apply the name cleaning on df_portfolio and do a fuzzy string matching
    # NB. since we assume: sum_on_cleaned_name=True, the forest and finance scores are already aggregated on the
    # cleaned name
    df_portfolio_financials['cleaned_name'] = df_portfolio_financials.name.apply(clean_company_name_new)

    # initialize empty lists for storing the variables of interest
    forest_and_finance_portfolio_amount = []
    forest_and_finance_portfolio_scores = []
    forest_and_finance_portfolio_flags = []

    print("Matching Forest & Finance data to portfolio...")
    # note; can be vectorized at some point
    for index, row in df_portfolio_financials.iterrows():

        match = find_best_match(row['cleaned_name'], forest_and_finance_scores['cleaned_name'],
                                score_cutoff=85)
        if match:
            valid_match = True
            # NB: this step is a manual step that needs to be done per portfolio!!
            # exclude false positives (manual inspection, so will not translate to other portfolios)
            if row['cleaned_name'] == 'hua nan financial':
                if match[0] == 'hana financial':
                    valid_match = False
            if row['cleaned_name'] == 'yuanta financial':
                if match[0] == 'hana financial':
                    valid_match = False
            if row['cleaned_name'] == 'reinet investments':
                if match[0] == 'nei investments':
                    valid_match = False
            if row['cleaned_name'] == 'sbi life insurance':
                if match[0] == 'nei investments':
                    valid_match = False
            if row['cleaned_name'] == 'ia financial':
                if match[0] == 'hana financial':
                    valid_match = False
        else:
            valid_match = False

        if valid_match:
            # if there was a valid match, check if the exposure was significant, add both the absolute amount,
            # as well as the min max score (without winsorizing; NB. this can be changed)
            # also add a flag: for we use a hard cut-off in millions of dollars (summing over the YOI)

            # Filter the DataFrame based on your condition
            filtered_ff_scores = forest_and_finance_scores[forest_and_finance_scores.cleaned_name == match[0]]
            forest_and_finance_portfolio_amount.append(float(filtered_ff_scores.AmountUSDMillions))
            forest_and_finance_portfolio_scores.append(float(filtered_ff_scores.original_MinMax_Scaled_Value))
            forest_and_finance_portfolio_flags.append(
                1.0 if float(filtered_ff_scores.AmountUSDMillions) >= FOREST_FINANCE_THRESHOLD else 0)

        else:
            forest_and_finance_portfolio_amount.append(0)
            forest_and_finance_portfolio_scores.append(0)
            forest_and_finance_portfolio_flags.append(0)

    # Drop cleaned_name column
    df_portfolio_financials = df_portfolio_financials.drop(columns=['cleaned_name'])

    df_portfolio_financials['forest_and_finance_amount'] = forest_and_finance_portfolio_amount
    df_portfolio_financials['forest_and_finance_score'] = forest_and_finance_portfolio_scores
    df_portfolio_financials['forest_and_finance_flag'] = forest_and_finance_portfolio_flags

    df_portfolio_nonfinancials['forest_and_finance_amount'] = 0
    df_portfolio_nonfinancials['forest_and_finance_score'] = 0
    df_portfolio_nonfinancials['forest_and_finance_flag'] = 0

    # Concatenate the dataframes back together
    df_portfolio = pd.concat([df_portfolio_financials, df_portfolio_nonfinancials], ignore_index=True)

    # Sorting the combined dataframe to maintain original order, if necessary
    df_portfolio.sort_index(inplace=True)

    # Print statement for quality control
    new_columns = set(df_portfolio.columns) - initial_columns
    print(f"Number of columns added: {len(new_columns)}")

    for column in new_columns:
        non_zero_count = (df_portfolio[column] != 0).sum()
        print(f"Number of non-zero values in '{column}': {non_zero_count}")

    return df_portfolio
