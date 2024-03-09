#!/bin/bash

# Move to the directory
cd $PWD/gan_protein_structural_requirements/data/raw/model1/test

for pathname in ./*.gz; do
    gzip -dc "$pathname" >"./$( basename "$pathname" .gz )"
    rm "$pathname"
done

cd ../train
for pathname in ./*.gz; do
    gzip -dc "$pathname" >"./$( basename "$pathname" .gz )"
    rm "$pathname"
done


# cd $PWD/gan_protein_structural_requirements/data/raw/model2/test_inhibitors

# for pathname in ./*.gz; do
#     gzip -dc "$pathname" >"./$( basename "$pathname" .gz )"
#     rm "$pathname"
# done

# cd $PWD/gan_protein_structural_requirements/data/raw/model2/test_enzymes

# for pathname in ./*.gz; do
#     gzip -dc "$pathname" >"./$( basename "$pathname" .gz )"
#     rm "$pathname"
# done

# cd $PWD/gan_protein_structural_requirements/data/raw/model2/train_inhibitors

# for pathname in ./*.gz; do
#     gzip -dc "$pathname" >"./$( basename "$pathname" .gz )"
#     rm "$pathname"
# done

# cd $PWD/gan_protein_structural_requirements/data/raw/model2/train_enzymes

# for pathname in ./*.gz; do
#     gzip -dc "$pathname" >"./$( basename "$pathname" .gz )"
#     rm "$pathname"
# done
