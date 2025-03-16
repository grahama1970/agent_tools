#!/bin/bash
# download_sample_data.sh
#
# Official Documentation:
# - wget: https://www.gnu.org/software/wget/manual/wget.html
#
# This script uses wget to recursively download the ArangoDB AQL documentation
# from https://docs.arangodb.com/stable/aql/ into the designated sample data directory.
# It is configured to avoid clobbering existing files.
#
# Usage:
#   ./download_sample_data.sh

# Define the root URL and the output directory relative to this script's location.
ROOT_URL="https://docs.arangodb.com/stable/aql/"
# The output directory is set relative to the scripts folder:
OUTPUT_DIR="../data/sample_data/arangodb_aql"

# Create the output directory if it does not exist
mkdir -p "${OUTPUT_DIR}"

# Run wget with appropriate parameters:
# --recursive           : Recursively download the entire site.
# --no-clobber         : Do not overwrite any existing files.
# --page-requisites    : Download all assets needed to display HTML.
# --html-extension     : Save files with a .html extension.
# --convert-links      : Convert links for local viewing.
# --restrict-file-names=windows : Ensure filenames are Windows-compatible.
# --domains docs.arangodb.com : Restrict downloads to this domain.
# --no-parent          : Do not ascend to the parent directory.
# -P <OUTPUT_DIR>      : Save all files into the specified output directory.
wget --recursive \
     --no-clobber \
     --page-requisites \
     --html-extension \
     --convert-links \
     --restrict-file-names=windows \
     --domains docs.arangodb.com \
     --no-parent \
     -P "${OUTPUT_DIR}" \
     "${ROOT_URL}"

echo "Download completed. Files saved in ${OUTPUT_DIR}"
