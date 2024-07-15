
# 🌳 Financial Deforestation Due Diligence

## Description

This repository complements the report "Making Deforestation Due Diligence Work in Practice - A Practical Methodology & Implementation Guidance for Financial Institutions" written by Climate & Company, in collaboration with AP2 and Global Canopy. [Read the report here](https://climateandcompany.org/publications/making-deforestation-due-diligence-work-in-practice). The Appendix in that document contains further detailed information on this repository.

The methodology has been developed to apply the guidance created by Global Canopy, Neural Alpha, and the Stockholm Environment Institute at scale. [Access the guidance document here](https://guidance.globalcanopy.org/further-guidance/due-diligence-towards-deforestation-free-finance/).

By default, this repository leverages publicly accessible information only. It processes the data in `portfolio_data.xlsx` located in `./data/input`, which contains core company information on the MSCI ACWI universe, and feeds it into various functions and models. Some code snippets that rely on proprietary data only contain placeholder scripts that need to be adjusted if the user wants to incorporate in-house proprietary data.

By default, the code produces an Excel file 📈  [df_output_open_source.xlsx](https://github.com/ClimateAndCompany/deforestation_free_finance/raw/main/data/output/df_output_open_source.xlsx) which contains relevant deforestation indicators for the MSCI ACWI universe. Keeping its limitations in mind, you can use this to get started. Note that the accuracy can be improved by incorporating your own proprietary data.


## Table of Contents

- [Installation](#-installation)
- [Usage](#-usage)
- [Folders Overview](#-folders-overview)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact Information](#-contact-information)
- [Disclaimer](#-disclaimer)
- [Changelog](#changelog)

## 🛠 Installation

Instructions on how to install and set up your project.

```bash
# Example for setting up the environment (we strongly recommend you to first create a dedicated venv)
git clone https://github.com/ClimateAndCompany/deforestation_free_finance.git
cd deforestation_free_finance
pip install . 
```

NB. due to package updates, requirements.txt will be updated. If issues arise, please download the
required packages manually.

## 🛠 Usage

- Replace data in `portfolio_data.xlsx` located in `./data/input` with your own portfolio data (if desired). It is also suggested to replace the NACE sector codes.
- The `run_dt1.py` and `run_dt2.py` files are the main files the user should use.
- Please change the path to your local environment in `filepaths.py`.
- While we suggest default settings in `user_input.py`, these values can be changed if the user is familiar with the methodology (see report).
- Users might encounter issues with downloading EXIOBASE data via the pymrio package. In that case, download EXIOBASE 3.8.2 via [zenodo](https://zenodo.org/records/5589597)


## 📁 Folders Overview

- 📊 `data`: Contains input data and pre-processed interim data.
- 📊 `generate`: Pure data generation not linked to a specific portfolio. Output in Python or Excel format.
- 📊 `prep`: Post-processing of generated data, linking it to the portfolio companies under scope.
- 📊 `apply`: Apply analysis and create flags containing meaningful information.

## 🤝 Contributing

We welcome feedback to improve our methodology, help in adding new datasets, and information on other potential use-cases. For implementation or methodological aspects, you can leave a comment or make a direct pull request on our GitHub repository. Alternatively, we can be reached by email.

## 📝 License

This code repository compiles relies on several existing sources that come with their own licensing terms. Here an overview:
- Asset-level data, Spatial Finance Initiative: CC BY 4.0 [link](https://www.cgfi.ac.uk/spatial-finance-initiative/);
- Asset-level data, Global Energy Monitor: CC BY 4.0 [link](https://globalenergymonitor.org/projects/global-coal-plant-tracker/download-data/);
- Asset-level data, Climate Trace: CC BY 4.0 [link](https://climatetrace.org/data) 
- SPOTT data: no license spotted [link](https://www.spott.org/)
- Forest 500 data:Creative Commons Attribution-NonCommercial 4.0, [link](https://forest500.org/terms-and-conditions/)
- Deforestation Action Tracke: no license spotted, [link](https://globalcanopy.org/what-we-do/corporate-performance/deforestation-action-tracker/)
- World Benchmarking Alliance, Nature: CC BY 4.0 [link](https://www.worldbenchmarkingalliance.org/nature-benchmark/) 
- EXIOBASE 3: CC-BY-SA 4.0 [link](https://www.exiobase.eu/index.php/terms-of-use)

Other aspects developed by Climate & Company fall under the CC BY NC SA 4.0 license (Attribution-NonCommercial-ShareAlike 4.0). See [link](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en).

## 📞 Contact Information

For questions or support, please contact the authors of 📃 [Making Deforestation Due Diligence Work - A Practical Methodology & Implementation Guidance for Financial Institutions](https://climateandcompany.org/publications/making-deforestation-due-diligence-work-in-practice):

💌 malte@climcom.org , marc@climcom.org , tycho@climcom.org . 

## 🔄 Disclaimer

**Note:** Various disclaimers apply. See [disclaimer.xlsx](https://github.com/ClimateAndCompany/deforestation_free_finance/raw/main/data/input/disclaimer.xlsx).

## Changelog

15th July 2024: fixed minor bugs (incorporated Pymrio update)
