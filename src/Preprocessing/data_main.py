from data_loading import download_multipl_e_raw
from data_preprocessing import process_multipl_e, process_instructcoder, process_codeedit

if __name__ == "__main__":
    print("Starting Dataset Processing Pipeline...")
    download_multipl_e_raw()
    process_multipl_e()
    process_instructcoder()
    process_codeedit()
