"""
Description:
    This script derives direct deforestation footprints from Pendrill data.

Update:
    Last updated in 06/2024
    - uses the data estimated by the DeDuCe model (Singh & Persson, 2024), which is an update of the results presented
        in Pendrill et al. (2019, 2022)

NOTES:
    - future users could decide to handle certain key 'missing' sectors separately. e.g. 'timber'
                (for more info on the methodology and its limitations, please read the original papers)
    - there are hardcoded estimates / assumptions made to derive a size measure for all the region, sector pairs;
                changes to the code could require users to adapt accordingly (potentially leading to some manual work)
"""

import country_converter as coco
import pandas as pd

from user_input import GDP_PROPORTIONS_DICT, CATTLE_SECTORS


### DEFINE HELPER FUNCTIONS ###


def pair_exists(pair, df):
    """
    Check if a given pair of values exists in the 'ISO' and 'FAO' columns of a DataFrame.

    Args:
        pair (tuple): A tuple (col1, col2) where col1 is checked against 'ISO' and col2 against 'FAO'.
        df (pandas.DataFrame): DataFrame containing 'ISO' and 'FAO' columns.

    Returns:
        bool: True if the pair exists, False otherwise.
    """
    col1, col2 = pair
    return ((df['ISO'] == col1) & (df['FAO'] == col2)).any()


def approximate_using_gdp(pair, gdp_data, gdp_proportions):
    """
    Approximate a value using GDP data and predefined GDP proportions.

    Args:
        pair (tuple): A tuple (ISO, FAO) where ISO is the country code and FAO is the item code.
        gdp_data (pandas.DataFrame): DataFrame containing 'ISO' and 'GDP' columns; NB. GDP should be in millions of $.
        gdp_proportions (dict): Dictionary mapping FAO item codes to GDP proportions.

    Returns:
        float: The approximated value calculated by multiplying the country's GDP by the FAO item's GDP proportion.
    """

    iso, fao = pair

    return float(gdp_data[gdp_data['ISO'] == iso]['GDP']) * gdp_proportions[fao]


def approximate_cattle_sector(pair, fao_data, cattle_sectors):
    """

    Approximate the value for the cattle sector by summing the values of related FAO items.

    Args:
        pair (tuple): A tuple (iso, fao) where iso is the country code and fao is the item code.
        fao_data (pandas.DataFrame): DataFrame containing 'ISO', 'FAO', and 'Value' columns.
        cattle_sectors (list): List of FAO item codes related to the cattle sector.

    Returns:
        float or bool: The sum of values for the related FAO items if matches are found, otherwise False.

    """
    iso, fao = pair

    matches = fao_data[(fao_data['ISO'] == iso) & (fao_data['FAO'].isin(cattle_sectors))]
    if len(matches) == 0:
        print('NO VALID AVERAGE POSSIBLE FOR PAIR: {}'.format(pair))
        return False
    else:
        return matches['Value'].sum()


### DEFINE THE GENERATE FUNCTION ###


def generate_direct_deforestation_footprints(path_to_deforestation_data,
                                             direct_attribution_year,
                                             path_to_faostat,
                                             path_to_gdp,
                                             path_to_faostat_production,
                                             path_to_faostat_production_price,
                                             path_to_deduce_mapping,
                                             hectare_cutoff,
                                             use_amortized):
    """
    Generate direct deforestation footprints.

    This function calculates the direct deforestation footprints for the region, sector pairs by integrating
    deforestation attribution data, FAOSTAT data, GDP data, production data, and production price data.
    The output is a DataFrame with the country ISO codes, FAO item codes, deforestation hectares, and the estimated
    economic size in millions of dollars.

    # NB. the values of agricultural production (main data source) are calculated based on production data of primary
    commodities from Production domain and producer prices from Prices domain. We will repeat some of that calculation
    manually to solve missing values.

    Args:
        path_to_deforestation_data (str): Path to the Excel file containing deforestation attribution data.
        direct_attribution_year (int): The year for which direct attribution data is relevant.
        path_to_faostat (str): Path to the Excel file containing FAOSTAT data.
        path_to_gdp (str): Path to the Excel file containing GDP data.
        path_to_faostat_production (str): Path to the Excel file containing FAOSTAT production data.
        path_to_faostat_production_price (str): Path to the Excel file containing FAOSTAT production price data.
        path_to_deduce_mapping (str): Path to the Excel file containing deduce mapping data.
        hectare_cutoff (float): Minimum deforestation hectares to consider for analysis.
        use_amortized (bool): If True, use amortized deforestation data. If False, use the data for just the specified
                                year.

    Returns:
        pandas.DataFrame: A DataFrame with columns ['ISO', 'FAO', 'Deforestation_hectares', 'Size', 'ISO2'] where
                          'Size' represents the estimated economic value in millions of dollars and 'ISO2' is the
                          ISO2 country code.
    """

    ### LOAD DATA ###

    concordance_data = pd.read_excel(path_to_deduce_mapping, sheet_name='lookup_adapted')
    # units are in hectares per million dollar
    attribution_data = pd.read_excel(path_to_deforestation_data, sheet_name='Deforestation attribution')
    # units are in thousands of dollars (value of agricultural production)
    faostat_data = pd.read_excel(path_to_faostat, sheet_name='faostat')
    # units are in dollars
    gdp_data = pd.read_excel(path_to_gdp, sheet_name='Data')
    # units are in the unit column, generally (tonnes)
    production_data = pd.read_excel(path_to_faostat_production, sheet_name='production_data')
    # units are annual value: price in USD per tonne
    production_price_data = pd.read_excel(path_to_faostat_production_price, sheet_name='production_price_data')

    # generate the necessary mappings to impute missing (ISO, FAO) pairs in the FAOSTAT data based on group averages
    mapping_fao_name_number = dict(zip(concordance_data['Item name'], concordance_data['FAO Item code']))
    mapping_fao_number_group = dict(zip(concordance_data['FAO Item code'], concordance_data['Group']))
    mapping_aggregated_by_group = concordance_data[['Group', 'FAO Item code']].groupby('Group')[
        'FAO Item code'].apply(list).reset_index(name='fao_item_codes')
    mapping_aggregated_by_group = dict(zip(mapping_aggregated_by_group['Group'],
                                           mapping_aggregated_by_group['fao_item_codes']))

    ### PROCESS ATTRIBUTION DATA ###

    # select only the relevant year (NB. current version of the data is only downloaded for a single year)
    attribution_data = attribution_data[attribution_data['Year'] == direct_attribution_year]

    # map the attribution data to the corresponding FAO code
    attribution_data['commodity_fao_number'] = attribution_data['Commodity'].map(mapping_fao_name_number)
    # select and rename the relevant columns
    # NB: note that one can choose to either use the unamortized or the amortized values
    if use_amortized:
        attribution_data = attribution_data[
            ['ISO', 'Deforestation risk, amortized (ha)', 'commodity_fao_number']]
        attribution_data = attribution_data.rename(columns={'commodity_fao_number': 'FAO',
                                                            'Deforestation risk, amortized (ha)': 'Deforestation_hectares'})
    else:
        attribution_data = attribution_data[
            ['ISO', 'Deforestation attribution, unamortized (ha)', 'commodity_fao_number']]
        attribution_data = attribution_data.rename(columns={'commodity_fao_number': 'FAO',
                                                            'Deforestation attribution, unamortized (ha)': 'Deforestation_hectares'})

    # aggregate on the (ISO, FAO) pairs in order to account for the few one-many mappings that occurred above
    # groupby and sum in order to account for many-one mappings (multiple commodities map to single FAO item code)
    attribution_data = attribution_data.groupby(['ISO', 'FAO'])['Deforestation_hectares'].sum().reset_index()

    # after this aggregation, remove deforestation attributions that are below the cutoff value specified by the user
    attribution_data = attribution_data[attribution_data['Deforestation_hectares'] > hectare_cutoff]

    iso_fao_attribution_triplets = list(zip(attribution_data['ISO'], attribution_data['FAO'],
                                            attribution_data['Deforestation_hectares']))

    # there are two areas of the attribution data not present in the GDP / faostat data because they are combinations
    # of regions: {'SDN and SSD', 'SRB and MNE'}; let's solve that

    ### PROCESS FAOSTAT DATA ###

    # NB. process the primary data source used to estimate the total size of each (ISO, FAO) pair in monetary terms,
    # i.e. the FAOSTAT value of agricultural production dataset

    # there are sometimes multiple rows that try to estimate the size of the pair; for some this is done in dollars,
    # for others this is done in "international dollars"; for some both: we average over the different estimates
    faostat_data = faostat_data.groupby(['Area Code (ISO3)', 'Item Code (FAO)'])['Value'].mean().reset_index()
    faostat_data['Value'] /= 1000.0

    # now, fix the faostat data to have combined (SDN, SSD) and (SRB, MNE) regions present as well
    faostat_data_sdn_ssd = faostat_data[faostat_data['Area Code (ISO3)'].isin(['SDN', 'SSD'])]
    faostat_data_sdn_ssd = faostat_data_sdn_ssd.groupby(['Item Code (FAO)'])['Value'].sum().reset_index()
    faostat_data_sdn_ssd['Area Code (ISO3)'] = 'SDN and SSD'

    faostat_data = pd.concat([faostat_data, faostat_data_sdn_ssd], ignore_index=True)

    faostat_data_srb_mne = faostat_data[faostat_data['Area Code (ISO3)'].isin(['SRB', 'MNE'])]
    faostat_data_srb_mne = faostat_data_srb_mne.groupby(['Item Code (FAO)'])['Value'].sum().reset_index()
    faostat_data_srb_mne['Area Code (ISO3)'] = 'SRB and MNE'

    faostat_data = pd.concat([faostat_data, faostat_data_srb_mne], ignore_index=True)
    faostat_data = faostat_data[['Area Code (ISO3)', 'Item Code (FAO)', 'Value']]
    faostat_data = faostat_data.rename(columns={'Area Code (ISO3)': 'ISO', 'Item Code (FAO)': 'FAO'})

    ### PROCESS GDP DATA ###

    # NB. for certain sectors (forest plantation, leather), FAOSTAT data is missing. we assume we can approximate
    # the size of the (ISO, FAO) pair using a fixed percentage of the GDP of the corresponding ISO. the fixed %s
    # are provided by the GDP_PROPORTIONS_DICT

    # add GDP data for the combined locations
    gdp_srb_mne = pd.DataFrame(
        {'Country Name': ['Serbia & Montenegro'], 'Country Code': ['SRB and MNE'], 'Series Name': ['XX'],
         'Series Code': 'XX', '2018 [YR2018]': [float(gdp_data[gdp_data['Country Code'] == 'SRB']['2018 [YR2018]'])
                                                + float(gdp_data[gdp_data['Country Code'] == 'MNE']['2018 [YR2018]'])]})
    gdp_sdn_ssd = pd.DataFrame(
        {'Country Name': ['Sudan & South Sudan'], 'Country Code': ['SDN and SSD'], 'Series Name': ['XX'],
         'Series Code': ['XX'], '2018 [YR2018]': [float(gdp_data[gdp_data['Country Code'] == 'SDN']['2018 [YR2018]'])
                                                  + float(
            gdp_data[gdp_data['Country Code'] == 'SSD']['2018 [YR2018]'])]})
    gdp_data = pd.concat([gdp_data, gdp_srb_mne], ignore_index=True)
    gdp_data = pd.concat([gdp_data, gdp_sdn_ssd], ignore_index=True)

    # now, pick out only the relevant columns in the GDP data and rename them
    gdp_data = gdp_data[['Country Code', '2018 [YR2018]']]
    gdp_data = gdp_data.rename(columns={'Country Code': 'ISO', '2018 [YR2018]': 'GDP'})
    # check that all the ISOs have a GDP associated, otherwise the analysis will not work
    if len(set(attribution_data.ISO).difference(set(gdp_data.ISO))) > 0:
        raise ValueError('please update the GDP data to include all the ISOs')

    # divide by one million to get the GDP in millions of dollars
    gdp_data['GDP'] /= 1000000.0

    ### PROCESS FAOSTAT PRODUCTION DATA ###

    # NB. in certain cases where faostat value data is missing we do have access to production data. in these cases
    # we can average over the producer prices for that commodity in the other countries to get an estimate

    # select only the correct year
    production_data = production_data[production_data['Year'] == direct_attribution_year]
    # select only data with the unit tonne. NB: you could do a unit conversion in the future if necessary
    production_data = production_data[production_data['Unit'] == 't']
    # select only the relevant columns and rename
    production_data = production_data[['Area Code (ISO3)', 'Item Code (FAO)', 'Value']]
    production_data = production_data.rename(columns={'Area Code (ISO3)': 'ISO', 'Item Code (FAO)': 'FAO'})

    ### PROCESS FAOSTAT PRODUCTION PRICE DATA ###

    # NB. in certain cases where faostat value data is missing we do have access to price data. in these cases
    # we need to manually find an estimate for the production (in tonnes)

    # select only the correct year
    production_price_data = production_price_data[production_price_data['Year'] == direct_attribution_year]
    # select only the relevant columns and rename
    production_price_data = production_price_data[['Area Code (ISO3)', 'Item Code (FAO)', 'Value']]
    production_price_data = production_price_data.rename(columns={'Area Code (ISO3)': 'ISO', 'Item Code (FAO)': 'FAO'})

    ### FULLY MANUAL DATA ###

    # NB. there are certain cases in which we need to fully manually estimate the size of the (ISO, FAO) pair
    # for 2018 this is the case for 14 pairs

    # now, in the main loop, we treat the different types of triplets differently. in all cases in the end we are
    # looking for a quadruple of the kind: (ISO, FAO, Ha deforestation, SIZE in millions of dollars), i.e. we are
    # merely estimating the size for each triplet

    sizes = []

    for i, triplet in enumerate(iso_fao_attribution_triplets):
        iso, fao, ha = triplet
        iso_fao_pair = (iso, fao)

        len_production_data = len(production_data[(production_data['ISO'] == iso) & (production_data['FAO'] == fao)])
        len_production_price_data = len(production_price_data[(production_price_data['ISO'] == iso)
                                                              & (production_price_data['FAO'] == fao)])

        # NB. line below is not always used (only if no direct match to faostat data)
        group_fao_items = mapping_aggregated_by_group[mapping_fao_number_group[fao]]

        # simplest case: we have FAOSTAT size information
        if pair_exists(iso_fao_pair, faostat_data):
            sizes.append(float(faostat_data[(faostat_data['ISO'] == iso) & (faostat_data['FAO'] == fao)]['Value']))
        # in the case the fao code is equal to leather or forest plantation, we estimate as fixed % of the ISO GDP
        elif fao in list(GDP_PROPORTIONS_DICT.keys()):
            sizes.append(approximate_using_gdp(iso_fao_pair, gdp_data, GDP_PROPORTIONS_DICT))
        # in the case the fao code is cattle meat, we sum the corresponding cattle and buffalo fao sectors
        elif fao == 1806:
            sum_over_cattle_sectors_value = approximate_cattle_sector(iso_fao_pair, faostat_data, CATTLE_SECTORS)
            if not sum_over_cattle_sectors_value:
                # NB. in the case of 2018, this is only the case for the ('GNQ', 1806) pair. The reason is that price
                # data is not present for 'GNQ'. However, there is production data available, so, let's rely on that
                # and take the average over the production prices for all the cattle related sectors. This logic would
                # have to be adapted if in future cases there is also no production data available.

                if len(production_data[(production_data['ISO'] == iso) &
                                       (production_data['FAO'].isin(CATTLE_SECTORS))]) == 0:
                    raise ValueError('no production faostat data available associated with the ISO, please solve')

                # rely on the production data for the ISO, and the average price per tonne over the other ISOs
                # for the FAO cattle sectors
                # average price per tonne
                average_price_per_tonne = production_price_data[
                    production_price_data['FAO'].isin(CATTLE_SECTORS)].Value.mean()
                # production has been filtered to be in units of tonnes also, so we can safely multiply
                sum_over_cattle_production_sectors = production_data[(production_data['ISO'] == iso) &
                                                                     (production_data['FAO'].isin(
                                                                         CATTLE_SECTORS))].Value.sum()

                sizes.append(average_price_per_tonne * sum_over_cattle_production_sectors / 1000000.0)
            else:
                sizes.append(sum_over_cattle_sectors_value)
        elif len(faostat_data[(faostat_data['ISO'] == iso) & (faostat_data['FAO'].isin(group_fao_items))]) != 0:
            # in this case there are matches in the same FAO group for the specific ISO. we will assume that
            # that data is representative for the missing ISO, FAO pair
            matches = faostat_data[(faostat_data['ISO'] == iso) & (faostat_data['FAO'].isin(group_fao_items))]
            sizes.append(matches['Value'].mean())
        elif len_production_data != 0:
            # in this case we have production data
            production_quantity = float(production_data[(production_data['ISO'] == iso) &
                                                        (production_data['FAO'] == fao)].Value)
            if len_production_price_data != 0:
                # in this case we have both production data (in tonnes) and production price data (USD/tonne)
                price_per_tonne = float(production_price_data[(production_price_data['ISO'] == iso) &
                                                              (production_price_data['FAO'] == fao)].Value)
                sizes.append(production_quantity * price_per_tonne / 1000000.0)
            else:
                # in this case, we do have production data, but we do not have price data; let's solve that by taking
                # the average price for the fao code over all the other regions if it exists
                price_matches_fao_code = production_price_data[production_price_data['FAO'] == fao].Value
                if len(price_matches_fao_code) != 0:
                    sizes.append(production_quantity * price_matches_fao_code.mean() / 1000000.0)
                else:
                    # in the case specifically for FAO code 777 (Hemp fibre and tow) there is no price information
                    # available in the entire price information dataset so in that case we need to solve it manually
                    if iso_fao_pair == ('DEU', 777):
                        # price per tonne seems to lie around 300 dollars
                        # Source: https://www.hempbenchmarks.com/hemp-market-insider/2023-wholesale-hemp-product-price-trends/)
                        sizes.append(production_quantity * 300.0 / 1000000.0)
                    else:
                        raise ValueError('check the assumptions for the new dataset being used')
        elif len_production_price_data != 0:
            # in this case we do have price data, but we do not have production size data, let's solve that manually
            price_per_tonne = float(production_price_data[(production_price_data['ISO'] == iso) &
                                                          (production_price_data['FAO'] == fao)].Value)
            # 459 = Chicory roots. In Belgium this seems to lie around 300.000 tonnes
            # Source: https://www.helgilibrary.com/indicators/chicory-root-production/
            # 677 = Hops (fresh and dried). In the EU: 26.500 Ha of Hops produces around 50.000 tonnes of hops.
            # In Belgium, this is around 181 Ha, or 181*(50K/26.5K)=341.51; Sources:
            # https://agriculture.ec.europa.eu/farming/crop-productions-and-plant-based-products/hops_en
            # https://www.belgischehop.be/en/beer-lover/belgian-hop#:~:text=23%20Belgian%20hop%20growers%20are,and%20expertise%20for%20many%20generations.
            # 773 = Flax fibre and tow. In Latvia the production is negligible; around 0.3 tonnes
            # Source: https://www.ceicdata.com/en/latvia/crop-production/crop-production-flax-fibre
            production_data_custom_values = {
                ('BEL', 459): 300000.0, ('BEL', 677): 341.51, ('LVA', 773): 0.3
            }
            if iso_fao_pair in list(production_data_custom_values.keys()):
                sizes.append(production_data_custom_values[iso_fao_pair] * price_per_tonne / 1000000.0)
            else:
                raise ValueError('check the assumptions for the new dataset being used')
        else:
            # in this case we have to estimate both separately, this is the case for 15 pairs

            manual_estimates_in_millions_usd = {
                ('BGR', 689): 1.787,  # dried chillies in Bulgaria: (based on faostat 2017; most recent year)
                ('CHL', 839): 0.1,
                # natural gum in Chile: this is likely even an overestimate; https://wits.worldbank.org/trade/comtrade/en/country/CHL/year/2019/tradeflow/Exports/partner/ALL/product/130190
                ('DNK', 711): 0.112,  # anise, etc. in Denmark: (based on faostat 2017; most recent year)
                ('GRC', 689): 0.676,  # dried chillies in Greece: (based on faostat 2017; most recent year)
                ('GRC', 711): 1.752,  # anise, etc. in Greece: (based on faostat 2017; most recent year)
                ('HUN', 723): 5.422,  # other spices in Hungary (based on faostat 2017; most recent year)
                ('IRL', 777): 0.02 * 300 * 179020 / 1000000.0,
                # hemp in Ireland; let's assume 2% of European production (179,020), at 300USD/tonne. https://agriculture.ec.europa.eu/farming/crop-productions-and-plant-based-products/hemp_en#:~:text=Hemp%20is%20a%20crop%20grown,(a%2084.3%25%20increase).
                ('LTU', 711): 9.768,  # anise, etc. in Lithuania: (based on faostat 2017; most recent year)
                ('LVA', 777): 0.02 * 300 * 179020 / 1000000.0,
                # hemp in Latvia; let's assume 2% of European production (179,020), at 300USD/tonne. https://agriculture.ec.europa.eu/farming/crop-productions-and-plant-based-products/hemp_en#:~:text=Hemp%20is%20a%20crop%20grown,(a%2084.3%25%20increase).
                ('NLD', 181): 0.566,  # broad beans in Netherlands: (based on faostat 2017; most recent year)
                ('NLD', 459): 53.899,  # chicory roots in Neterlands: (based on faostat 2017; most recent year)
                ('NLD', 711): 0.737,  # anise, etc. in Netherlands: (based on faostat 2017; most recent year)
                ('SGP', 27): 0.05 * 0.005 * float(gdp_data[gdp_data['ISO'] == 'SGP'].GDP),
                # rice in singapore, only 0.5% of GDP is agriculture, rice is only a fraction of that; let's assume 5%. https://en.wikipedia.org/wiki/Agriculture_in_Singapore
                ('SVK', 777): 0.02 * 300 * 179020 / 1000000.0,
                # hemp in Slovakia; let's assume 2% of European production (179,020), at 300USD/tonne. https://agriculture.ec.europa.eu/farming/crop-productions-and-plant-based-products/hemp_en#:~:text=Hemp%20is%20a%20crop%20grown,(a%2084.3%25%20increase).
                ('SVN', 157): 45.0 * 6965 / 1000000.0
                # sugar beets in Slovania, based on 2023 stats from the national statistics agenct STAT.si, and assumed price per tonne of 45 dollars. https://pxweb.stat.si/SiStatData/pxweb/en/Data/-/1502402S.px/table/tableViewLayout2/
            }
            if iso_fao_pair in list(manual_estimates_in_millions_usd.keys()):
                sizes.append(manual_estimates_in_millions_usd[iso_fao_pair])
            else:
                raise ValueError('check the assumptions for the new dataset being used')

    iso_fao_attribution_size_quadruplets = pd.DataFrame(list(zip(attribution_data['ISO'],
                                                                 attribution_data['FAO'],
                                                                 attribution_data['Deforestation_hectares'],
                                                                 sizes)),
                                                        columns=['ISO', 'FAO', 'Deforestation_hectares', 'Size'])

    # add ISO2
    iso_fao_attribution_size_quadruplets['ISO2'] = coco.convert(
        names=iso_fao_attribution_size_quadruplets.ISO, to='ISO2')

    return iso_fao_attribution_size_quadruplets
