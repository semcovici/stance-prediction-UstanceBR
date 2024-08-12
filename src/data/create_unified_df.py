import os
import pandas as pd
from tqdm import tqdm
import logging

tqdm.pandas()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
PATH_RAW_DATA = 'data/raw/'
PATH_PROCESSED_DATA = 'data/processed/'
USERS_PATH = os.path.join(PATH_PROCESSED_DATA, "{split}_r3_{target}_top_mentioned_timelines_processed.csv")
TMT_PATH = os.path.join(PATH_PROCESSED_DATA, 'r3_{target}_{split}_users_processed.csv')
TARGETS = ['bo', 'cl', 'co', 'gl', 'ig', 'lu']
SPLITS = ['train', 'test']

def read_data(file_path):
    """Read CSV file and handle exceptions."""
    try:
        return pd.read_csv(file_path, sep=';', encoding='utf-8-sig', index_col=0)
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return pd.DataFrame()

def process_and_merge_data(target, split):
    """Process and merge user and TMT data."""
    users_file_path = USERS_PATH.format(split=split, target=target)
    tmt_file_path = TMT_PATH.format(split=split, target=target)

    data_users = read_data(users_file_path)
    data_tmt = read_data(tmt_file_path)

    if data_users.empty or data_tmt.empty:
        logging.warning(f"Skipping {target}-{split} due to read errors.")
        return

    data = data_users.merge(data_tmt, on=['User_ID', 'Polarity'], how='outer')

    # fill null values with an empty string 
    data.fillna('', inplace=True)
    
    # create column with texts and timeline concateneted
    data["concat_Texts_Timeline"] = data.Texts + " # " + data.Timeline

    output_file_path = os.path.join(PATH_PROCESSED_DATA, f"{split}_unified_processed_df_{target}_processed.csv")
    data.to_csv(output_file_path, sep=';', encoding='utf-8-sig', index=False)
    logging.info(f"Saved unified data to {output_file_path}")

def main():
    """Main function to process data for all targets and splits."""
    for target in tqdm(TARGETS, disable=False):
        for split in SPLITS:
            process_and_merge_data(target, split)

if __name__ == "__main__":
    main()
