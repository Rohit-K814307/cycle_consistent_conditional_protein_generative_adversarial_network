#!/bin/sh

#chmod everything
chmod +x ./gan_protein_structural_requirements/scripts/data_download.sh
chmod +x ./gan_protein_structural_requirements/scripts/esm_download.sh
chmod +x ./gan_protein_structural_requirements/scripts/uncompress.sh
chmod +x ./gan_protein_structural_requirements/scripts/get_data_ids.sh

#make important dirs
mkdir -p $PWD/gan_protein_structural_requirements/data/raw/model1/train
mkdir -p $PWD/gan_protein_structural_requirements/data/raw/model1/test
mkdir -p $PWD/gan_protein_structural_requirements/data/raw/model2/train_targets
mkdir -p $PWD/gan_protein_structural_requirements/data/raw/model2/train_inhibitors
mkdir -p $PWD/gan_protein_structural_requirements/data/raw/model2/test_targets
mkdir -p $PWD/gan_protein_structural_requirements/data/raw/model2/test_inhibitors

#download esm
#./gan_protein_structural_requirements/scripts/esm_download.sh

#get data ids
./gan_protein_structural_requirements/scripts/get_data_ids.sh

#download
./gan_protein_structural_requirements/scripts/data_download.sh -f ./gan_protein_structural_requirements/data/raw/model1/train_ids.txt -o ./gan_protein_structural_requirements/data/raw/model1/train -p
./gan_protein_structural_requirements/scripts/data_download.sh -f ./gan_protein_structural_requirements/data/raw/model1/test_ids.txt -o ./gan_protein_structural_requirements/data/raw/model1/test -p

# ./gan_protein_structural_requirements/scripts/data_download.sh -f ./gan_protein_structural_requirements/data/raw/model2/train_inhibitor_ids.txt -o ./gan_protein_structural_requirements/data/raw/model2/train_inhibitors -p
# ./gan_protein_structural_requirements/scripts/data_download.sh -f ./gan_protein_structural_requirements/data/raw/model2/train_enzyme_ids.txt -o ./gan_protein_structural_requirements/data/raw/model2/train_targets -p
# ./gan_protein_structural_requirements/scripts/data_download.sh -f ./gan_protein_structural_requirements/data/raw/model2/test_inhibitor_ids.txt -o ./gan_protein_structural_requirements/data/raw/model2/test_inhibitors -p
# ./gan_protein_structural_requirements/scripts/data_download.sh -f ./gan_protein_structural_requirements/data/raw/model2/test_enzyme_ids.txt -o ./gan_protein_structural_requirements/data/raw/model2/test_targets -p

#uncompress the data
./gan_protein_structural_requirements/scripts/uncompress.sh

#save the data to pytorch to make it easier

mkdir -p $PWD/gan_protein_structural_requirements/data/save/model1
mkdir -p $PWD/gan_protein_structural_requirements/data/save/model2

python -m gan_protein_structural_requirements.data.save_datasets 0 300

echo "Repo Setup Complete"