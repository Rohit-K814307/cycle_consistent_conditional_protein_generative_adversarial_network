#!/bin/bash

# Move to the directory
cd $PWD/gan_protein_structural_requirements/data/raw/batch_1_data

for pathname in ./*.gz; do
    gzip -dc "$pathname" >"./$( basename "$pathname" .gz )"
    rm "$pathname"
done

