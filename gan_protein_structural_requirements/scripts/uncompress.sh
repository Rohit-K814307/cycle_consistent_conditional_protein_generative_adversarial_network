#!/bin/bash

# Move to the directory
cd $PWD/gan_protein_structural_requirements/data/raw/batch_1_data

# Loop through .gz files
for gz_file in *.gz; do
    # Get the filename without the .gz extension
    uncompressed_file="${gz_file%.gz}"

    # Decompress the .gz file and replace it with the uncompressed version
    gzip -d -c "$gz_file" > "$uncompressed_file"

    # Optionally, you may want to remove the original .gz file
    rm "$gz_file"

    echo "Uncompressed $gz_file to $uncompressed_file"
done

echo "Setup script completed."
