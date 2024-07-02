## Before running this file make sure you ran the following:
# 1. run_decision_tree1.py
# 2. run_decision_tree2.py

# import packages
import pandas as pd
import os
from filepaths import PATH_TO_OUTPUT_FOLDER, PATH_TO_VARIABLE_DESCRIPTIONS

df_variable_description = pd.read_excel(PATH_TO_VARIABLE_DESCRIPTIONS, sheet_name='variable_description')

## Add dataframe from DT1 & DT2 to the final Excel file

# Columns for basic company variables
columns_basic_company_variables = [
    'company_name', 'identifier', 'country_name', 'nace_code', 'nace_desc', 'number_of_pairs_asset',
    'number_of_pairs_country_sector'
]

# Load the data with the specified columns and order
df_columns_basic_company_variables = pd.read_excel(os.path.join(PATH_TO_OUTPUT_FOLDER, 'output_final_dt1.xlsx'),
                                                   usecols=columns_basic_company_variables)

# Columns for DT1 variables
columns_dt1 = [
    'identifier', 'sector_direct_score', 'sector_indirect_score', 'io_supply_chain_score', 'flag_forest500',
    'direct_attribution_score', 'asset_impact_assignment_count_asset', 'dt1_bucket', 'dt1_score'
]
# Load the data with the specified columns and order
df_columns_dt1 = pd.read_excel(os.path.join(PATH_TO_OUTPUT_FOLDER, 'output_final_dt1.xlsx'),
                               usecols=columns_dt1)

# Columns for DT2 variables
columns_dt2 = ['identifier', 'forest500_score', 'human_rights_policy_in_place', 'deforestation_policy_in_place',
               'strength_hr_policy', 'strength_deforestation_policy', 'positive_flag', 'dt2_bucket']
# Load the data with the specified columns and order
df_columns_dt2 = pd.read_excel(os.path.join(PATH_TO_OUTPUT_FOLDER, 'output_final_dt2.xlsx'),
                              usecols=columns_dt2)

columns_open_source = [
    "identifier",
    "wba_NAT.B01",
    "wba_NAT.B06",
    "wba_NAT.C02",
    "wba_NAT.C05",
    "wba_NAT.C07",
    "sbti_long_term_target",
    "sbti_near_term_target",
    "spott_sust_policy_score",
    "spott_landbank_score",
    "spott_cert_standards_score",
    "spott_def_biodiv_score",
    "spott_hcv_hcs_score",
    "spott_soils_fire_score",
    "spott_community_land_labour_score",
    "spott_smallholders_suppliers_score",
    "spott_gov_grievance_score",
    "spott_rspo_member",
    "tnfd_early_adopter",
    "tnfd_positive_flag",
    "wba_positive_flag",
    "sbti_positive_flag",
    "spott_positive_flag",
]
# Load the data with the specified columns and order
df_columns_open_source = pd.read_excel(os.path.join(PATH_TO_OUTPUT_FOLDER, 'output_final_dt2.xlsx'),
                              usecols=columns_open_source)

# rename certain open source columns
df_columns_open_source = df_columns_open_source.rename(columns={"wba_NAT.B01": 'wba_impacts_on_nature_assessment',
    "wba_NAT.B06": 'wba_ecosystem_restoration',
    "wba_NAT.C05": 'wba_commitment_to_respect_human_rights',
    "wba_NAT.C07": 'wba_identifying_human_rights_risks_and_impacts',
    "wba_NAT.C02": 'wba_indigenous_peoples_rights'})


# Merge the three dfs for basic company variables, dt1 and dt2
df_output = pd.merge(df_columns_basic_company_variables, df_columns_dt1, on='identifier')
df_output = pd.merge(df_output, df_columns_dt2, on='identifier')
df_output = pd.merge(df_output, df_columns_open_source, on='identifier')


# # Insert empty rows at the specified positions
# insert_rows_variable_description = [0, 13, 27, 36]
# for row in insert_rows_variable_description:
#     empty_row = pd.DataFrame([[''] * len(df_variable_description.columns)], columns=df_variable_description.columns)
#     df_variable_description = pd.concat([df_variable_description.iloc[:row], empty_row, df_variable_description.iloc[row:]]).reset_index(drop=True)

# Save the processed data to a new Excel file with headers in bold
with pd.ExcelWriter(os.path.join(PATH_TO_OUTPUT_FOLDER, 'df_output_open_source.xlsx'), engine='xlsxwriter') as writer:
    df_variable_description.to_excel(writer, sheet_name='variable_description', index=False)
    # Create an empty DataFrame for the empty sheet
    empty_df = pd.DataFrame()
    empty_df.to_excel(writer, sheet_name='>>', index=False)
    df_output.to_excel(writer, sheet_name='data_output', startrow=1, index=False)
    
    # Add the disclaimer sheet
    df_disclaimer = pd.read_excel('data/input/disclaimer.xlsx')
    df_disclaimer.to_excel(writer, sheet_name='Disclaimer', index=False)

    # Get the xlsxwriter workbook and worksheet objects
    workbook = writer.book
    worksheet_variable_description = writer.sheets['variable_description']
    worksheet_data_output = writer.sheets['data_output']

    # Define a format for the header cells
    header_format = workbook.add_format({'bold': True})

    # Write the headers with the defined format for the 'variable_description' sheet
    for col_num, value in enumerate(df_variable_description.columns.values):
        worksheet_variable_description.write(0, col_num, value, header_format)

    # Write the headers with the defined format for the 'data_output' sheet
    for col_num, value in enumerate(df_output.columns.values):
        worksheet_data_output.write(0, col_num, value, header_format)

    # Define a format for the merged cell
    merge_format_grey = workbook.add_format({
        'bold': True,
        'align': 'center',
        'valign': 'vcenter',
        'bg_color': '#D3D3D3'
    })
    merge_format_orange = workbook.add_format({
        'bold': True,
        'align': 'center',
        'valign': 'vcenter',
        'bg_color': '#FFE4C4'
    })
    merge_format_yellow = workbook.add_format({
        'bold': True,
        'align': 'center',
        'valign': 'vcenter',
        'bg_color': '#FFFACD'
    })
    merge_format_green = workbook.add_format({
        'bold': True,
        'align': 'center',
        'valign': 'vcenter',
        'bg_color': '#9ACD32'
    })

    worksheet_data_output.merge_range(
        'A1:G1', 'Basic Company Variables', merge_format_grey
    )
    worksheet_data_output.merge_range(
        'H1:O1', 'Variables linked to DT1 (EXPOSURE)', merge_format_orange
    )
    worksheet_data_output.merge_range(
        'P1:V1', 'Variables linked to DT2 (POLICY)', merge_format_yellow
    )
    worksheet_data_output.merge_range(
        'W1:AR1', 'Further open source variables we collected', merge_format_green
    )

    # Set the column width to be twice as wide as the default
    default_width = 8.43  # Default width in Excel
    worksheet_data_output.set_column('A:AR', default_width * 2)
