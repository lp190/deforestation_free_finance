"""
Description:
    This script processes the Forest & Finance dataset. It processes the data to
    calculate the total financing amounts by banks over a specified number of years, applies winsorization to handle
    outliers, and normalizes the financing amounts using MinMax scaling. The script outputs a DataFrame containing
    the company name, forest finance absolute amounts, and their respective scores.

Update:
    Last updated in Q2 2024
    Source: https://forestsandfinance.org/data/

Output:
    A DataFrame containing the company name, forest finance absolute amounts, and normalized scores.

NOTES:
    - The methodology could be improved by making it commodity/country-specific.
    - Cutoffs for winsorizing are currently hardcoded (95%/99%).
"""

# Import necessary libraries
import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import MinMaxScaler

from utils import clean_company_name


def generate_forest_finance_scores(path_to_forest_finance, ff_yoi, ff_truncation, sum_on_cleaned_name=True):
    """
    This function generates the forest finance scores. It is based on the forest_finance.xlsx file.

    Args:
        path_to_forest_finance (str): path to original xlsx forest and finance file
        ff_yoi (int): year of interest
        ff_truncation (int): number of years you look back in time
        sum_on_cleaned_name (bool): whether to sum the values on the cleaned company name

    Returns:
        (pd.dataframe): company name, forest and finance absolute amount, as well as scores
    """

    forest_finance_scores = pd.read_excel(path_to_forest_finance, sheet_name=0)

    # generate list of years counting backwards from ff_yoi:
    years_of_interest = list(range(ff_yoi, ff_yoi - ff_truncation - 1, -1))

    # select only values that fall within years_of_interest
    forest_finance_scores = forest_finance_scores[forest_finance_scores['Year'].isin(years_of_interest)]

    # select only the columns: bank, amount, and rename the columns:
    forest_finance_scores = forest_finance_scores[['Bank', 'AmountUSDMillions']]

    # Manual data cleaning:
    # --> rename "iA Financial Group" to "IA Financial Group" (since this is clearly the same entity)
    forest_finance_scores['Bank'] = forest_finance_scores['Bank'].replace('iA Financial Group', 'IA Financial Group')

    # Rename columns
    forest_finance_scores.rename(columns={'Bank': 'company_name'}, inplace=True)

    if sum_on_cleaned_name:
        # Clean company names via applying the clean_company_name function
        # NB. make sure that this is aligned with the clean_company_name function to do the matching to portfolio
        # company names in the apply file
        forest_finance_scores['cleaned_name'] = forest_finance_scores['company_name'].apply(clean_company_name)
        # drop original name
        forest_finance_scores = forest_finance_scores.drop(columns='company_name')
        # Take the sum
        forest_finance_scores = forest_finance_scores.groupby(['cleaned_name']).sum().reset_index()
    else:
        raise ValueError("Manual matching required")

    # also add the 95% and 99%
    forest_finance_scores['AmountUSDMillions_top95_winsorized'] = winsorize(forest_finance_scores['AmountUSDMillions'],
                                                                            limits=[0, 0.05])
    forest_finance_scores['AmountUSDMillions_top99_winsorized'] = winsorize(forest_finance_scores['AmountUSDMillions'],
                                                                            limits=[0, 0.01])
    scaler = MinMaxScaler()

    forest_finance_scores['original_MinMax_Scaled_Value'] = \
        scaler.fit_transform(pd.DataFrame(forest_finance_scores['AmountUSDMillions']))

    forest_finance_scores['winsorized_95_MinMax_Scaled_Value'] = \
        scaler.fit_transform(pd.DataFrame(forest_finance_scores['AmountUSDMillions_top95_winsorized']))

    forest_finance_scores['winsorized_99_MinMax_Scaled_Value'] = \
        scaler.fit_transform(pd.DataFrame(forest_finance_scores['AmountUSDMillions_top99_winsorized']))

    return forest_finance_scores
