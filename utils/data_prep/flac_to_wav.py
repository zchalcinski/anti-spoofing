import os
import argparse
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import time

def convert_flac_to_wav_recursive(source_dir, dest_dir, overwrite=False):
    """
    Recursively finds FLAC files in source_dir and converts them to
    16-bit PCM WAV files in dest_dir, preserving the directory structure.
    """
    source_base = Path(source_dir).resolve() # Get absolute path
    dest_base = Path(dest_dir).resolve()   # Get absolute path

    if not source_base.is_dir():
        print(f"Error: Source directory '{source_base}' not found or is not a directory.")
        return

    # Create destination base directory if it doesn't exist
    dest_base.mkdir(parents=True, exist_ok=True)
    print(f"Source directory: {source_base}")
    print(f"Destination directory: {dest_base}")
    print(f"Overwrite existing WAV files: {overwrite}")

    # Find all FLAC files recursively (case-insensitive)
    print("Finding FLAC files...")
    flac_files = list(source_base.rglob('*.flac')) + list(source_base.rglob('*.FLAC'))

    if not flac_files:
        print("No FLAC files found.")
        return

    print(f"Found {len(flac_files)} FLAC files. Starting conversion...")

    converted_count = 0
    skipped_count = 0
    error_count = 0
    start_time = time.time()

    # Process files with a progress bar
    for flac_path in tqdm(flac_files, desc="Converting"):
        try:
            # Determine relative path to maintain structure
            relative_path = flac_path.relative_to(source_base)

            # Create corresponding WAV path in the destination directory
            wav_relative_path = relative_path.with_suffix('.wav')
            wav_full_path = dest_base / wav_relative_path

            # Create parent directories for the WAV file if they don't exist
            wav_full_path.parent.mkdir(parents=True, exist_ok=True)

            # Check if WAV file already exists and handle overwrite logic
            if not overwrite and wav_full_path.exists():
                # Optional: print skipped message
                # print(f"Skipping '{flac_path.name}': WAV file already exists at '{wav_full_path}'")
                skipped_count += 1
                continue

            # Read FLAC file
            data, samplerate = sf.read(flac_path, dtype='float32') # Read as float32 for processing

            # Write WAV file as 16-bit PCM
            # Common subtypes: 'PCM_16', 'PCM_24', 'PCM_32', 'FLOAT', 'DOUBLE'
            # PCM_16 is widely compatible.
            sf.write(wav_full_path, data, samplerate, subtype='PCM_16')
            converted_count += 1

        except Exception as e:
            error_count += 1
            print(f"\nError converting file '{flac_path}': {e}") # Print newline to avoid messing up tqdm

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\n--- Conversion Summary ---")
    print(f"Conversion process took {elapsed_time:.2f} seconds.")
    print(f"Successfully converted: {converted_count} files")
    if not overwrite:
        print(f"Skipped (already exist): {skipped_count} files")
    print(f"Errors during conversion: {error_count} files")

# --- Main execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FLAC files to WAV recursively.")
    parser.add_argument("source_directory", help="Directory containing the source FLAC files.")
    parser.add_argument("destination_directory", help="Directory where WAV files will be saved (structure preserved).")
    parser.add_argument("-o", "--overwrite", action="store_true", help="Overwrite existing WAV files in the destination directory.")

    args = parser.parse_args()

    convert_flac_to_wav_recursive(args.source_directory, args.destination_directory, args.overwrite)

    print("\nScript finished.")