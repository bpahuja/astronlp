import lmdb
import os
import argparse
from tqdm import tqdm

def merge_lmdbs(source_db_paths, target_db_path, overwrite_keys=True):
    """
    Merges multiple LMDB databases into a single new one.

    Args:
        source_db_paths (list): A list of paths to the source LMDB databases.
        target_db_path (str): The path for the new, merged LMDB database.
        overwrite_keys (bool): If True, keys from later databases will overwrite
                               earlier ones in case of collision. If False,
                               the first-written key will be kept.
    """
    if os.path.exists(target_db_path):
        raise ValueError(f"Target database path already exists: {target_db_path}. "
                         "Please provide a path for a new database.")

    # Set a very large map size to avoid "MapFullError"
    # This is virtual address space, not initial disk usage.
    map_size = 1024 * 1024 * 1024 * 10  # 100 GB, adjust as needed

    # Create the target database environment
    target_env = lmdb.open(target_db_path, map_size=map_size, writemap=True)
    
    total_keys_written = 0

    for db_path in source_db_paths:
        if not os.path.exists(db_path):
            print(f"Warning: Source database not found, skipping: {db_path}")
            continue

        print(f"Processing source database: {db_path}")
        source_env = lmdb.open(db_path, readonly=True, lock=False)

        with source_env.begin() as source_txn, target_env.begin(write=True) as target_txn:
            # Use a cursor for efficient iteration
            cursor = source_txn.cursor()
            
            # Get the total number of entries for the progress bar
            num_entries = source_txn.stat()['entries']
            
            pbar = tqdm(cursor.iternext(keys=True, values=True), total=num_entries, desc=f"Copying from {os.path.basename(db_path)}")
            
            for key, value in pbar:
                # The core operation: put the key-value pair into the target DB
                # The `overwrite` parameter handles key collisions.
                # True (default): last write wins.
                # False: first write wins (skip if key exists).
                if target_txn.put(key, value, overwrite=overwrite_keys):
                    total_keys_written += 1

        source_env.close()

    print("\nMerge complete.")
    print(f"Total unique keys written to target database: {total_keys_written}")
    
    # You can also get the final count from the new DB stat
    with target_env.begin() as final_txn:
        final_count = final_txn.stat()['entries']
        print(f"Final entry count in '{target_db_path}': {final_count}")

    target_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple LMDB databases into one.")
    parser.add_argument(
        "source_dbs",
        nargs='+',
        help="List of source LMDB database paths to merge."
    )
    parser.add_argument(
        "--target_db",
        required=True,
        help="Path for the new, merged LMDB database."
    )
    parser.add_argument(
        "--skip_duplicates",
        action='store_true',
        help="If set, do not overwrite existing keys (first-write-wins). Default is to overwrite (last-write-wins)."
    )

    args = parser.parse_args()

    # If --skip_duplicates is True, we want overwrite=False
    should_overwrite = not args.skip_duplicates

    merge_lmdbs(args.source_dbs, args.target_db, overwrite_keys=should_overwrite)

    # --- Example Command-Line Usage ---
    # python merge_script.py /path/to/db1 /path/to/db2 --target_db /path/to/merged_db
    #
    # To keep the first value in case of key collision:
    # python merge_script.py /path/to/db1 /path/to/db2 --target_db /path/to/merged_db --skip_duplicates
