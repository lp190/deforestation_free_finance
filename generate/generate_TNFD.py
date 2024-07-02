"""
Description:
    This script extracts and processes data on TNFD early adopters from the TNFD website. It collects data from multiple 
    pages, cleans the company names, and generates a DataFrame indicating which companies are TNFD early adopters.

Update:
    Last updated in June 2024

Output:
    A DataFrame containing the cleaned company names and a flag indicating TNFD early adopters.
    
NOTES:
- Note that if the website structure changes, the script may need to be updated.  
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup


def extract_data_from_page(url):
    """
    Extract data from an HTML table on a webpage.

    This function sends a GET request to the specified URL, parses the HTML content to find a table with the class
    'is-responsive', and extracts the data from the table rows. The header row of the table is skipped, and the
    remaining rows are processed to extract the text content from each cell.

    Args:
        url (str): The URL of the webpage containing the table.

    Returns:
        list: A list of lists, where each inner list contains the text content of a row's cells. If no table is found,
              an empty list is returned.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', class_='is-responsive')
    if table is None:
        return []  # Return an empty list if no table is found
    rows = table.find_all('tr')[1:]  # Skip the header row

    data = []
    for row in rows:
        cols = row.find_all('td')
        cols = [ele.text.strip() for ele in cols]
        data.append(cols)
    return data


def clean_company_names(company_name):
    """
    Clean company names by removing specific substrings and extra whitespace.

    Args:
        company_name (str): The original company name string.

    Returns:
        str: The cleaned company name string with 'Early Adopter' removed and extra whitespace trimmed.
    """

    return company_name.split('Early Adopter')[0].strip()


def generate_TNFD():
    """
    Generates a DataFrame of TNFD early adopters by extracting data from multiple pages on the TNFD website,
    cleaning the company names, and adding a flag indicating TNFD early adopters.

    The function navigates through multiple pages of the TNFD early adopters list, collects the relevant data,
    cleans the company names to remove any extraneous text, and consolidates the data into a single DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the cleaned company names and a flag indicating TNFD early adopters.
    """
    print("Generating TNFD data via web scraping...")

    base_url = 'https://tnfd.global/engage/inaugural-tnfd-early-adopters/'
    page = 1
    all_data = []
    max_pages = 100  # Set a high limit to the number of pages to prevent infinite loops

    while page <= max_pages:
        if page == 1:
            url = base_url
        else:
            url = f'{base_url}?sf_paged={page}'

        data = extract_data_from_page(url)
        if not data:  # If no data is found, break the loop
            break

        all_data.extend(data)
        page += 1

    # Create a DataFrame
    df = pd.DataFrame(all_data, columns=['Organisation and Jurisdiction HQ',
                                         'TNFD-aligned disclosure(s) by financial year',
                                         'Sector Classification (SASB)',
                                         'Type of Institution'])

    # Rename the column
    df = df[['Organisation and Jurisdiction HQ']]
    df = df.rename(columns={'Organisation and Jurisdiction HQ': 'company_name'})

    # Clean the company names
    df['company_name'] = df['company_name'].apply(clean_company_names)
    df['tnfd_early_adopter'] = 1

    return df
