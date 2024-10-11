import logging
import os
from datetime import datetime
from config import LOG_FOLDER_PATH

def setup_logging(app_name):
    # Ensure the log folder exists
    os.makedirs(LOG_FOLDER_PATH, exist_ok=True)

    # Get the current timestamp to add to the log filename
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # Configure logging with timestamp in the log file name
    log_filename = os.path.join(LOG_FOLDER_PATH, f'{app_name}_{current_time}.log')

    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,  # Set logging level to INFO
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='a'  # Append to the log file
    )

    # Log to both file and console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info(f"Logging configurado para: {app_name}.")
