from extraction_and_treatment.data_extraction import extraction

def main():
    # Call the extraction function to fetch data from both APIs and save it to CSV
    print("Starting data extraction process...")
    extraction()
    print("Data extraction completed and saved to CSV files.")

if __name__ == "__main__":
    main()
