"""
Description:
    This script prepares and matches Forest 500 data with the specified portfolio data.
    It links the Forest 500 financial institutions and companies to their respective identifier,
    using both manual and automated matching processes. The script cleans and processes the data,
    performs string matching, and handles potential false positives.

Output:
    The script outputs a CSV file 'forest500_matches.csv' containing the merged data with Forest 500 controversies 
    scores and flags indicating valid matches.

Outline of code:
    1) Load and process Forest 500 data
    2) Perform manual matching if specified
    3) Clean and standardize company names
    4) Perform fuzzy string matching
    5) Merge and export the data
    
NOTES:
 - There are some incorrect matches; the function could be improved by users in the future; for instance by using more
    advanced nlp-based mapping, https://docs.dedupe.io/en/latest/, https://github.com/Living-with-machines/DeezyMatch
"""

import os.path

import country_converter as coco
import pandas as pd

from filepaths import PATH_TO_F500_FIRMS, PATH_TO_F500_FIS, PATH_TO_F500_MANUAL_MATCHES, \
    PATH_TO_OUTPUT_FOLDER
from utils import find_best_match, clean_company_name


def prep_forest500(df_portfolio, columns_comps, columns_fis, fuzzy_match_cutoff, false_positive_list,
                   manual_matches):
    """
    Prepare and match Forest 500 controversies data with an input DataFrame of index weights. This has to be done
    in order to link the Forest500 FI's and companies to their respective identifier (Permid).

    Args:
        df_portfolio (pd.DataFrame): DataFrame containing index weights information with at least a name and permid column.
        columns_comps (tuple): List of company names to be used from the Forest 500 companies CSV file.
        columns_fis (tuple): List of FI names to be used from the Forest 500 financial institutions CSV file.
        fuzzy_match_cutoff (float): The cutoff score for the fuzzy string matching algorithm.
        false_positive_list (tuple): List of company names identified as false positives.
        manual_matches (bool): If True, use manual matching for permids; otherwise, perform automated matching.

    Returns:
        pd.DataFrame: Merged DataFrame containing index weights along with Forest 500 controversies data,
                      including Forest 500 total score and a flag indicating a valid match.
    """

    # Load F500 Companies
    controv_forest500_comps = pd.read_csv(PATH_TO_F500_FIRMS, usecols=columns_comps)
    controv_forest500_comps = controv_forest500_comps.drop_duplicates()  # Drop duplicate rows
    controv_forest500_comps.columns = controv_forest500_comps.columns.str.lower().str.strip()
    controv_forest500_comps.rename(columns={'company': 'name_forest500',
                                            'total score /100': 'forest500_total_score',
                                            'hq': 'hq_country'}, inplace=True)

    # Load Forest500 Financial Institutions
    controv_forest500_fis = pd.read_csv(PATH_TO_F500_FIS, usecols=columns_fis)
    controv_forest500_fis = controv_forest500_fis.drop_duplicates()  # Drop duplicate rows
    controv_forest500_fis.columns = controv_forest500_fis.columns.str.lower().str.strip()
    controv_forest500_fis.rename(columns={'fi name': 'name_forest500',
                                          'total score / 100': 'forest500_total_score',
                                          'fi headquarters': 'hq_country'}, inplace=True)

    # Third Forest500 permid matches
    forest500_matches = pd.read_csv(PATH_TO_F500_MANUAL_MATCHES)
    forest500_matches.rename(columns={'forest500_name': 'name_forest500'},
                             inplace=True)  # Rename the "fi_name" column to "name"

    # Concatenate the two datasets
    merged_controv_forest500 = pd.concat([controv_forest500_comps, controv_forest500_fis], ignore_index=True)

    if manual_matches:
        # merge the merged_controv_forest500 to the permid list
        merged_controv_forest500_permid = pd.merge(forest500_matches, merged_controv_forest500, on='name_forest500',
                                                   how='left')

        merged_controv_forest500_permid['name_forest500'] = merged_controv_forest500_permid[
            'name_forest500'].str.lower()
        merged_controv_forest500_permid['hq_country'] = merged_controv_forest500_permid['hq_country'].str.lower()
        forest500_countries = list(merged_controv_forest500_permid['hq_country'])
        merged_controv_forest500_permid['iso2'] = [i.lower() for i in
                                                   coco.convert(names=forest500_countries, to='ISO2')]

        # Add forest 500 flag
        merged_controv_forest500_permid['flag_forest500'] = 1

        merged_controv_forest500_permid.to_csv(
            os.path.join(PATH_TO_OUTPUT_FOLDER, 'internal_data/forest500_matches.csv'))
        return merged_controv_forest500_permid
    else:
        merged_controv_forest500['name'] = merged_controv_forest500['name'].str.lower()
        forest500_names = list(merged_controv_forest500['name'])
        # now, clean the company names (removing inc, etc), so you get more/better hits
        # (NB. effectiveness is not extensively tested)
        merged_controv_forest500['cleaned_name'] = [clean_company_name(i) for i in forest500_names]
        merged_controv_forest500['hq_country'] = merged_controv_forest500['hq_country'].str.lower()
        forest500_countries = list(merged_controv_forest500['hq_country'])
        merged_controv_forest500['iso2'] = [i.lower() for i in coco.convert(names=forest500_countries, to='ISO2')]

        print('string matching now, this takes some time')

        # Create a new DataFrame to store the merged data
        # note: a 0 for the forest500_total_score means that no valid match was found
        merged_data = pd.DataFrame(columns=['permid'] + ['forest500_total_score'] + ['flag_forest500'])

        # Now, for each of the companies in index weights, try to match to the names in the forest500 dataset
        for index, row in df_portfolio.iterrows():
            controversy_forest500_row = row.copy()[['permid']]
            issuer = row['name'].lower().strip()
            # cleaned issuer
            cleaned_issuer = clean_company_name(issuer)
            iso2_domicil = row['country_domicil'].lower().strip()
            iso2_country = row['country_iso'].lower().strip()
            iso2_country_permid = row['country_permid'].lower().strip()
            # multiple isos are possible, so let's check for matches to any to increase the chance of a match
            isos_company = list(set([iso2_domicil, iso2_country, iso2_country_permid]))

            # in order to improve the matching, for each company in the dataset, only try to match with the datapoints in
            # forest500 that have an ISO2 matching with any of those associated with the company
            filtered_df = merged_controv_forest500[merged_controv_forest500['iso2'].isin(isos_company)]

            if len(filtered_df) > 0:
                match_issuer = find_best_match(cleaned_issuer, filtered_df['cleaned_name'],
                                               score_cutoff=fuzzy_match_cutoff)
                if match_issuer:
                    if cleaned_issuer in false_positive_list:
                        # set score and flag to 0; no VALID match (match was manually identified false positive)
                        controversy_forest500_row['forest500_total_score'] = 0.0
                        controversy_forest500_row['flag_forest500'] = 0.0
                    else:
                        # set for that company (row) the forest500 score equal to the score of the match
                        controversy_forest500_row['forest500_total_score'] = \
                            float(filtered_df[filtered_df.index == match_issuer[2]]['forest500_total_score'])
                        # and set the forest500 flag to 1
                        controversy_forest500_row['flag_forest500'] = 1.0
                else:
                    # set score and flag to 0; no valid match (based on no match according to fuzzy string match on name)
                    controversy_forest500_row['forest500_total_score'] = 0.0
                    controversy_forest500_row['flag_forest500'] = 0.0
            else:
                # set score and flag to 0; no valid match (based on ISO2)
                controversy_forest500_row['forest500_total_score'] = 0.0
                controversy_forest500_row['flag_forest500'] = 0.0

            merged_data = pd.concat([merged_data, pd.concat(
                [pd.DataFrame(controversy_forest500_row).T], axis=1)], ignore_index=True)

        merged_data.to_csv(os.path.join(PATH_TO_OUTPUT_FOLDER, 'internal_data/forest500_matches.csv'))

        return merged_data
