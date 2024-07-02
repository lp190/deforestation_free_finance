"""
Description:
    Derives insights from deforestation hotspot data.
    
Update:
    Last updated in 06/2024
    Source: https://data.globalforestwatch.org/datasets/gfw::emerging-hot-spots-2023/about
    
NOTES:
- Consider to incorporate asset time series data (i.e. asset might not have been there / owned by the company in year X)
    this would start to become an issue if the hotspot data and ownership data concern very different points in time
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from shapely.geometry import Point
from shapely.ops import nearest_points


def prep_gfw_hotspot(df_portfolio,
                     distance_threshold,
                     type_of_asset,
                     impact_dict,
                     path_to_deforestation_hotspots,
                     path_to_disaggregated_assets,
                     path_to_subsidiaries):
    """
    Prepare and analyze geographic data to identify the impact of hotspots on assets.

    NB. one could improve the functionality by:
        * making the distance threshold depend on the type of asset under consideration
        * weighing the score differently depending on the type of asset (e.g. a factory within a certain threshold
                of a hotspot could be rated differently than a subsidiary).
        * taking into account the hierarchy structure in a more nuanced way (e.g. weigh subsidiaries further removed
                from the parent entity lower).

    Args:
        df_portfolio (DataFrame): the main portfolio dataframe.
        distance_threshold (float): distance threshold; assets within this distance are considered to be exposed to the
                                    deforestation associated with the hotspot.
        type_of_asset (string): either 'asset' or 'subsidiary' is currently supported, otherwise ValueError.
        impact_dict (dict): Dictionary mapping hotspot types to their impact weights.
        path_to_deforestation_hotspots (str): file path to hotspot geodata.
        path_to_disaggregated_assets (str): file path to disggregated assets data.
        path_to_subsidiaries (str): file path to subsidiaries with associated (lat,lon) information.

    Returns:
        DataFrame: A DataFrame containing the final results with hotspot-proximity information aggregated
                    to company level.
    """

    # Load .GeoJson file into a GeoDataFrame
    gdf_multipolygons = gpd.read_file(path_to_deforestation_hotspots)

    if type_of_asset == 'asset':
        asset_locations = pd.read_csv(path_to_disaggregated_assets)[['permid', 'longitude', 'latitude']]
    elif type_of_asset == 'subsidiary':
        asset_locations = pd.read_csv(path_to_subsidiaries)[['isin_par', 'latitude', 'longitude']]
        asset_locations['latitude'] = asset_locations['latitude'].astype(float)
        asset_locations['longitude'] = asset_locations['longitude'].astype(float)
        # remove duplicates
        asset_locations = asset_locations.drop_duplicates()
        # drop nans
        asset_locations = asset_locations.dropna()
        # change column name to align with assets
        asset_locations = asset_locations.rename(columns={"isin_par": "isin"})
    else:
        raise ValueError('not yet supported')

    identifier_assets = asset_locations['permid']
    asset_coordinates_series = gpd.GeoSeries(asset_locations.apply(
        lambda row: Point(row['longitude'], row['latitude']), axis=1),
        crs="EPSG:4326")
    asset_coordinates_series = asset_coordinates_series.to_crs(gdf_multipolygons.crs)

    # Initialize DataFrame to collect results
    results_df = pd.DataFrame(columns=['coordinate_index', 'closest_polygon_index', 'distance'])

    # Iterate over each point to calculate distances to all polygons and find the closest one
    for idx, point in asset_coordinates_series.items():
        # NB: for now, just take the closest match, but in the future we could take 'the worst one' within
        # the 'range' of the asset
        distances1 = gdf_multipolygons.geometry.distance(point)
        closest_polygon_index1 = distances1.idxmin()
        closest_distance = distances1.min()
        closest_polygon_index = closest_polygon_index1
        nearest_point = nearest_points(point,
                                       gdf_multipolygons[
                                           gdf_multipolygons['FID'] == closest_polygon_index].geometry)[1]

        # !! geodisc takes (lat, long) tuples, so the opposite way around

        # NB. there are some points that are empty; it happens in about 2% of the cases
        if len(nearest_point) == 0:
            print('encountered an empty geometry')
        else:
            nearest_point_distance = geodesic(
                (point.y, point.x),
                (float(nearest_point.y), float(nearest_point.x))
            ).kilometers
            # Append results to the DataFrame
            new_row = pd.DataFrame([{
                'coordinate_index': idx,
                'identifier': identifier_assets[idx],
                'closest_polygon_index': closest_polygon_index,
                'impact_weight': impact_dict[closest_polygon_index],
                'distance': closest_distance,
                'nearest_point': nearest_point,
                'nearest_point_distance': nearest_point_distance
            }])
            results_df = pd.concat([results_df, new_row], ignore_index=True)

        # Print every 500th iteration
        if (idx + 1) % 500 == 0:
            print(f'{idx + 1} out of {len(asset_coordinates_series)} iterations')

    # now we have computed the distances between each of the assets and the closest polygon
    # now check which asset is within the threshold away from a polygon

    results_df['asset_impact_assignment' + '_' + type_of_asset] = np.where(
        results_df['nearest_point_distance'] <= distance_threshold,
        results_df['impact_weight'],  # Value to assign if condition is True
        0  # Value to assign if condition is False (can change this as needed)
    )

    results_df['asset_impact_assignment_count' + '_' + type_of_asset] = np.where(
        results_df['nearest_point_distance'] <= distance_threshold,
        results_df['impact_weight'],  # Value to assign if condition is True
        0  # Value to assign if condition is False (can change this as needed)
    )

    results_df['asset_impact_assignment_count_unweighted' + '_' + type_of_asset] = np.where(
        results_df['nearest_point_distance'] <= distance_threshold,
        1,  # Value to assign if condition is True
        0  # Value to assign if condition is False (can change this as needed)
    )

    aggregated_results = results_df.groupby('identifier')['asset_impact_assignment' + '_' + type_of_asset].agg(
        ['mean']).rename(columns={'mean': 'average_asset_impact_assignment' + '_' + type_of_asset})
    aggregated_results_count = results_df.groupby('identifier')[
        'asset_impact_assignment_count' + '_' + type_of_asset].agg(['sum']).rename(
        columns={'sum': 'total_asset_impact_assignment' + '_' + type_of_asset})
    aggregated_results_count_unweighted = results_df.groupby('identifier')[
        'asset_impact_assignment_count_unweighted' + '_' + type_of_asset].agg(['sum']).rename(
        columns={'sum': 'asset_impact_assignment_count' + '_' + type_of_asset})

    # NB. this could be moved elsewhere to improve readability (ensure same format for the identifier)
    # ---- start of ad-hoc fix
    aggregated_results = aggregated_results.reset_index()
    aggregated_results['identifier'] = aggregated_results['identifier'].astype(str).str.replace('.0', '')

    aggregated_results_count = aggregated_results_count.reset_index()
    aggregated_results_count['identifier'] = aggregated_results_count['identifier'].astype(str).str.replace('.0', '')

    aggregated_results_count_unweighted = aggregated_results_count_unweighted.reset_index()
    aggregated_results_count_unweighted['identifier'] = aggregated_results_count_unweighted['identifier'].astype(
        str).str.replace('.0', '')
    # ---- end of ad-hoc fix

    identifier_df = df_portfolio[['identifier']]
    final_results = identifier_df.merge(aggregated_results, on='identifier', how='left')
    final_results.fillna(0, inplace=True)

    final_results = final_results.merge(aggregated_results_count, on='identifier', how='left')
    final_results.fillna(0, inplace=True)

    final_results = final_results.merge(aggregated_results_count_unweighted, on='identifier', how='left')
    final_results.fillna(0, inplace=True)

    return final_results
