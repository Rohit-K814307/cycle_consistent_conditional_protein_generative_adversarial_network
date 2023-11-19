#!/bin/sh


#chmod everything
chmod +x ./gan_protein_structural_requirements/scripts/data_download.sh
chmod +x ./gan_protein_structural_requirements/scripts/esm_download.sh
chmod +x ./gan_protein_structural_requirements/scripts/uncompress.sh

#download esm
./gan_protein_structural_requirements/scripts/esm_download.sh

#make dir for batch 1 and download
mkdir -p $PWD/gan_protein_structural_requirements/data/raw/batch_1_data
./gan_protein_structural_requirements/scripts/data_download.sh -f ./gan_protein_structural_requirements/scripts/list_file.txt -o ./gan_protein_structural_requirements/data/raw/batch_1_data -p
#uncompress the data
./gan_protein_structural_requirements/scripts/uncompress.sh