from config import SERIES_IDS, IPEADATA_SERIES_CODES, BCB_ENDPOINT, IPEADATA_ENDPOINT, CSV_SAVE_PATHS, EXTRACTION_CSV_FOLDER_PATHS
from extraction_and_treatment.BCBDataExtractor import BCBDataExtractor
from extraction_and_treatment.IpeadataExtractor import IpeadataExtractor

import os

def extraction():
    # Create output directory if it doesn't exist
    os.makedirs(EXTRACTION_CSV_FOLDER_PATHS, exist_ok=True)

    # Iterate over each series in SERIES_IDS and fetch the data
    for series_name, series_id in SERIES_IDS.items():
        # Instantiate the BCB data extractor
        bcb_extractor = BCBDataExtractor(series_id=series_id, series_name=series_name, base_url=BCB_ENDPOINT)
        
        # Fetch and process the data
        bcb_data = bcb_extractor.get_processed_data()
        
        # Dynamically create CSV file path using the series name
        csv_filename = EXTRACTION_CSV_FOLDER_PATHS + CSV_SAVE_PATHS['bcb'].format(series_name)
        
        # Save the data to CSV
        bcb_data.to_csv(csv_filename, index=False)
        print(f"BCB data for {series_name} saved to '{csv_filename}'")
    
    # Example: Fetch and process PIB data from Ipeadata
    for series_name, series_code in IPEADATA_SERIES_CODES.items():
        # Instantiate the Ipeadata extractor
        ipeadata_extractor = IpeadataExtractor(series_code=series_code, base_url=IPEADATA_ENDPOINT)
        
        # Fetch and process the PIB data
        pib_data = ipeadata_extractor.get_processed_data()
        
        # Dynamically create CSV file path using the series name
        csv_filename = EXTRACTION_CSV_FOLDER_PATHS + CSV_SAVE_PATHS['ipeadata'].format(series_name)
        
        # Save PIB data to CSV
        pib_data.to_csv(csv_filename, index=False)
        print(f"Ipeadata data for {series_name} saved to '{csv_filename}'")
