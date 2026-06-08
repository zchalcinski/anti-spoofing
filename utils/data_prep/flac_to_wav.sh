#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
# set -e # Optional: uncomment if you want script to stop on first ffmpeg error

# --- Configuration & Argument Parsing ---
FORCE_OVERWRITE=0 # Default: do not overwrite

# Simple argument parsing
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -f|--force)
      FORCE_OVERWRITE=1
      shift # past argument
      ;;
    *)
      # Assume positional arguments are source and destination
      if [[ -z "$SOURCE_DIR" ]]; then
        SOURCE_DIR="$1"
      elif [[ -z "$DEST_DIR" ]]; then
        DEST_DIR="$1"
      else
        echo "Unknown argument: $1"
        echo "Usage: $0 [-f|--force] <source_flac_dir> <destination_wav_dir>"
        exit 1
      fi
      shift # past argument
      ;;
  esac
done

# Check if required arguments are provided
if [[ -z "$SOURCE_DIR" ]] || [[ -z "$DEST_DIR" ]]; then
  echo "Error: Source and destination directories are required."
  echo "Usage: $0 [-f|--force] <source_flac_dir> <destination_wav_dir>"
  exit 1
fi

# Resolve to absolute paths for safety
SOURCE_DIR=$(realpath "$SOURCE_DIR")
DEST_DIR=$(realpath "$DEST_DIR")

# Check if source directory exists
if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Error: Source directory '$SOURCE_DIR' not found."
  exit 1
fi

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"
echo "Source directory:      $SOURCE_DIR"
echo "Destination directory: $DEST_DIR"
if [[ "$FORCE_OVERWRITE" -eq 1 ]]; then
  echo "Overwrite enabled:   Yes"
  FFMPEG_OVERWRITE_FLAG="-y"
else
  echo "Overwrite enabled:   No (use -f to force)"
  FFMPEG_OVERWRITE_FLAG="-n" # Do not overwrite
fi
echo "Starting conversion..."

# --- Find and Convert Files ---
SUCCESS_COUNT=0
SKIPPED_COUNT=0
ERROR_COUNT=0

# Use find to get list of files, print0/read -d for safety with special characters
find "$SOURCE_DIR" -type f -iname "*.flac" -print0 | while IFS= read -r -d $'\0' flac_file; do
  # Calculate relative path from source directory
  relative_path="${flac_file#$SOURCE_DIR/}"
  # Construct base path for destination (without extension)
  dest_path_base="$DEST_DIR/$relative_path"
  # Construct final WAV filename (replace any extension with .wav)
  wav_file="${dest_path_base%.*}.wav"
  # Get the directory part of the destination path
  dest_subdir=$(dirname "$wav_file")

  # Create the destination subdirectory if it doesn't exist
  mkdir -p "$dest_subdir"

  # --- Run ffmpeg ---
  echo "Converting: '$relative_path' -> '${wav_file#$DEST_DIR/}'"
  # -loglevel error: only show errors
  # -acodec pcm_s16le: standard 16-bit WAV codec
  ffmpeg -i "$flac_file" -acodec pcm_s16le "$FFMPEG_OVERWRITE_FLAG" -loglevel error "$wav_file"

  # Check ffmpeg exit status
  if [[ $? -eq 0 ]]; then
    # Check if ffmpeg actually wrote the file (it might not if -n was used and file existed)
    if [[ -f "$wav_file" ]]; then
      ((SUCCESS_COUNT++))
    else
      ((SKIPPED_COUNT++))
      # echo "Skipped (exists): $relative_path" # Optional verbose skip message
    fi
  else
    echo "ERROR converting '$flac_file'" >&2 # Print error message to stderr
    ((ERROR_COUNT++))
  fi

done

# --- Summary ---
echo "----------------------------------"
echo "Conversion Summary:"
echo "Successfully converted: $SUCCESS_COUNT"
if [[ "$FORCE_OVERWRITE" -eq 0 ]]; then
  echo "Skipped (already exist): $SKIPPED_COUNT"
fi
echo "Errors encountered:     $ERROR_COUNT"
echo "----------------------------------"
echo "Script finished."

# Check if there were errors and exit accordingly
if [[ $ERROR_COUNT -gt 0 ]]; then
  exit 1
else
  exit 0
fi
