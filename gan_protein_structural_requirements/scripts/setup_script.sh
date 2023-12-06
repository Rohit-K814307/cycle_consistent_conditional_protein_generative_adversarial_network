#!/bin/sh

#chmod everything
chmod +x ./gan_protein_structural_requirements/scripts/data_download.sh
chmod +x ./gan_protein_structural_requirements/scripts/esm_download.sh
chmod +x ./gan_protein_structural_requirements/scripts/uncompress.sh
chmod +x ./gan_protein_structural_requirements/scripts/get_data_ids.sh

#make important dirs
mkdir -p $PWD/gan_protein_structural_requirements/data/raw/train
mkdir -p $PWD/gan_protein_structural_requirements/data/raw/test

#download esm
#./gan_protein_structural_requirements/scripts/esm_download.sh

#get data ids
./gan_protein_structural_requirements/scripts/get_data_ids.sh

#download
./gan_protein_structural_requirements/scripts/data_download.sh -f ./gan_protein_structural_requirements/data/raw/train_ids.txt -o ./gan_protein_structural_requirements/data/raw/train -p
./gan_protein_structural_requirements/scripts/data_download.sh -f ./gan_protein_structural_requirements/data/raw/test_ids.txt -o ./gan_protein_structural_requirements/data/raw/test -p
#uncompress the data
./gan_protein_structural_requirements/scripts/uncompress.sh

echo "Repo Setup Complete"