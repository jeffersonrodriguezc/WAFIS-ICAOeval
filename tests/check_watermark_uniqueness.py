# -*- coding: utf-8 -*-
import sqlite3
import os
import argparse
from pathlib import Path

def check_uniqueness_and_count(db_path: str, bpp: int = 1) -> None:
    """
    Connects to the watermark database, reads all watermarks,
    and verifies if they are unique and counts the total.
    """
    # Convert the input database path string to a Path object for consistent handling.
    db_path = Path(db_path)
    # Check if the specified database file actually exists at the given path.
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at: {db_path}")
        # Print an error message if the database file does not exist.
        return
        # Exit the function early if the file is not found.

    conn = None
    # Initialize the database connection variable to None.
    try:
        # Establish a connection to the SQLite database file.
        conn = sqlite3.connect(db_path)
        # Create a cursor object, which allows you to execute SQL commands within the connection.
        cursor = conn.cursor()

        # Get all watermarks from the 'watermarks' table.
        cursor.execute("SELECT watermark_data FROM watermarks")
        # Fetch all results of the executed query. This returns a list of tuples,
        # where each tuple represents a row and contains the watermark data.
        all_watermarks = cursor.fetchall() # Retrieves a list of tuples, e.g., [('0101...',), ('1010...',)]

        # Check if any watermarks were retrieved.
        if not all_watermarks:
            print("No watermarks found in the database.")
            return

        # Get the total number of watermark entries retrieved from the database.
        total_entries = len(all_watermarks)
        # Initialize an empty set. A set automatically stores only unique elements,
        # making it perfect for checking for duplicates.
        unique_watermarks_set = set()

        # Iterate through each retrieved watermark and add its string representation to the set.
        # This process implicitly checks for uniqueness as sets do not allow duplicates.
        for row in all_watermarks:
            watermark_str = row[0] # The watermark is the first (and only) element in the tuple for each row.
            unique_watermarks_set.add(watermark_str)
            # Add the watermark string to the set. If the string is already present, it will not be added again.
        
        # Get the count of unique watermarks by checking the size of the set.
        unique_count = len(unique_watermarks_set)

        print(f"\n--- Watermark Verification Results ---")
        print(f"Database analyzed: {db_path}")
        print(f"Total watermarks registered: {total_entries}")
        print(f"Number of UNIQUE watermarks found: {unique_count}")

        # Compare the total number of entries with the count of unique entries.
        if total_entries == unique_count:
            # If they match, all watermarks are unique.
            print("\n✅ All watermarks are UNIQUE! The script functioned correctly in this aspect.")
        else:
            # If they don't match, duplicates were found.
            print(f"\n❌ WARNING! {total_entries - unique_count} duplicate watermarks were found.")
            # Indicate a potential problem with the watermark generation logic.
            print("This might indicate an issue in the watermark generation logic,")
            # Emphasize the statistical improbability of random collisions for this watermark size.
            print("although it is extremely improbable for this watermark size.")
        
        # Optional: Verify the length of a sample watermark.
        # This assumes all watermarks should have the same expected length.
        if total_entries > 0:
            # Get the length of the first watermark string as a sample.
            sample_watermark_length = len(all_watermarks[0][0])
            #print(all_watermarks[0][0])
            print(f"Length of a sample watermark: {sample_watermark_length} bits.")
            
            # Define the expected length based on your generation parameters (message_n * message_l).
            EXPECTED_LENGTH = 4096 * 16 * bpp # Based on 256x256 @ 1 bpp (64*64*16)
            # Check if the sample watermark's length matches the expected length.
            if sample_watermark_length == EXPECTED_LENGTH:
                print(f"✅ Watermark length matches the expected length ({EXPECTED_LENGTH} bits).")
            else:
                # Alert if the length does not match, indicating a possible discrepancy in generation parameters.
                print(f"❌ Watermark length ({sample_watermark_length}) DOES NOT match the expected length ({EXPECTED_LENGTH} bits).")


    except sqlite3.Error as e:
        # Catch any exceptions that occur during SQLite database operations.
        print(f"Error accessing the SQLite database: {e}")
        # Print the specific error message.
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verifies the uniqueness and count of watermarks in the database.')
    # Add an argument for the database file path.
    parser.add_argument('--db_path', type=str, 
                        # Set a default path for convenience.
                        # The 'r' prefix creates a raw string, preventing backslashes from being
                        # interpreted as escape sequences, which is crucial for Windows paths.
                        default=r'..\datasets\facelab_london\processed\watermarks\watermarks_BBP_2_131072_13107.db',
                        help='Full path to the SQLite watermark database file.')
    parser.add_argument('--bpp', type=int, default=1,
                        help='Bits per pixel (bpp) for the watermark, assuming bpp=1.')
    
    # Parse the command-line arguments provided by the user.
    args = parser.parse_args()
    # Call the main function to perform the uniqueness and count checks using the provided DB path.
    check_uniqueness_and_count(args.db_path, bpp=args.bpp)
