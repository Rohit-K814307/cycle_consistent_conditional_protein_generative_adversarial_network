#!/bin/bash

echo "Model 1 data & ids"

wget -O gan_protein_structural_requirements/data/raw/model1/uniprot_id_data.tsv "https://rest.uniprot.org/uniprotkb/search?fields=accession%2Clength%2Cid&format=tsv&query=%28%28database%3ADrugBank%29+AND+%28reviewed%3Atrue%29%29+AND+%28model_organism%3A9606%29+AND+%28proteins_with%3A1%29+AND+%28annotation_score%3A5%29+AND+%28length%3A%5B1+TO+200%5D%29&size=500"

touch gan_protein_structural_requirements/data/raw/model1/train_ids.txt
touch gan_protein_structural_requirements/data/raw/model1/test_ids.txt

tsv_file="gan_protein_structural_requirements/data/raw/model1/uniprot_id_data.tsv"
output_file_train="gan_protein_structural_requirements/data/raw/model1/train_ids.txt"
output_file_test="gan_protein_structural_requirements/data/raw/model1/test_ids.txt"
column_number=1

# Skip the first row (header), extract values, and split between two output files
tail -n +2 "$tsv_file" | cut -f"$column_number" | awk 'BEGIN {srand()} {if (rand() < 0.7) printf("%s,", $0) >> "'"$output_file_train"'"; else printf("%s,", $0) >> "'"$output_file_test"'"}'

# Remove trailing commas from both files
sed -i -e 's/.$//'  "$output_file_train"
sed -i -e 's/.$//' "$output_file_test"
rm gan_protein_structural_requirements/data/raw/model1/train_ids.txt-e
rm gan_protein_structural_requirements/data/raw/model1/test_ids.txt-e


#################### train ##################

# turn ids into pdb ids
train_ids_data="$(cat gan_protein_structural_requirements/data/raw/model1/train_ids.txt)"

jobId="$(curl -s --form 'from="UniProtKB_AC-ID"' \
     --form 'to="PDB"' \
     --form "ids="$train_ids_data"" \
     https://rest.uniprot.org/idmapping/run | jq -r '.jobId')"

read -p "Pause Time 1 second" -t 1

json_data="$(curl -s "https://rest.uniprot.org/idmapping/stream/$jobId")"

from_values=()

# Extract values of "to" key and concatenate into a comma-separated string, skipping repeats
result=""
for entry in $(echo "$json_data" | jq -c '.results[]'); do
     from_value=$(echo "$entry" | jq -r '.from')
     to_value=$(echo "$entry" | jq -r '.to')

     if [[ ! " ${from_values[@]} " =~ " $from_value " ]]; then
          result="${result}${to_value},"
          # Add this "from" value to the array
          from_values+=("$from_value")
     fi
done

result="${result%,}"

rm gan_protein_structural_requirements/data/raw/model1/train_ids.txt
echo "$result" > gan_protein_structural_requirements/data/raw/model1/train_ids.txt


#################### test ##################

# turn ids into pdb ids
train_ids_data="$(cat gan_protein_structural_requirements/data/raw/model1/test_ids.txt)"

jobId="$(curl -s --form 'from="UniProtKB_AC-ID"' \
     --form 'to="PDB"' \
     --form "ids="$train_ids_data"" \
     https://rest.uniprot.org/idmapping/run | jq -r '.jobId')"

read -p "Pause Time 1 second" -t 1

json_data="$(curl -s "https://rest.uniprot.org/idmapping/stream/$jobId")"

from_values=()

# Extract values of "to" key and concatenate into a comma-separated string, skipping repeats
result=""
for entry in $(echo "$json_data" | jq -c '.results[]'); do
     from_value=$(echo "$entry" | jq -r '.from')
     to_value=$(echo "$entry" | jq -r '.to')

     if [[ ! " ${from_values[@]} " =~ " $from_value " ]]; then
          result="${result}${to_value},"
          # Add this "from" value to the array
          from_values+=("$from_value")
     fi
done

result="${result%,}"


rm gan_protein_structural_requirements/data/raw/model1/test_ids.txt
echo "$result" > gan_protein_structural_requirements/data/raw/model1/test_ids.txt


rm gan_protein_structural_requirements/data/raw/model1/uniprot_id_data.tsv
echo "Extraction complete. Results saved to $output_file_train and $output_file_test"



#########################################################################################

# wget -O gan_protein_structural_requirements/data/raw/model2/uniprot_id_data.tsv "https://rest.uniprot.org/uniprotkb/search?fields=accession%2Cid%2Clength%2Ccc_interaction&format=tsv&query=%28%28database%3AMEROPS%29+AND+%28reviewed%3Atrue%29%29+AND+%28annotation_score%3A5%29&size=500"

# python -m gan_protein_structural_requirements.data.model2_data_read_ids


# #################### train ##################

# # turn ids into pdb ids
# train_inhibitor_ids_data="$(cat gan_protein_structural_requirements/data/raw/model2/train_inhibitor_ids.txt)"

# jobId="$(curl -s --form 'from="UniProtKB_AC-ID"' \
#      --form 'to="PDB"' \
#      --form "ids="$train_inhibitor_ids_data"" \
#      https://rest.uniprot.org/idmapping/run | jq -r '.jobId')"

# read -p "Pause Time 1 second" -t 1

# json_data="$(curl -s "https://rest.uniprot.org/idmapping/stream/$jobId")"

# from_values=()

# # Extract values of "to" key and concatenate into a comma-separated string, skipping repeats
# result=""
# for entry in $(echo "$json_data" | jq -c '.results[]'); do
#      from_value=$(echo "$entry" | jq -r '.from')
#      to_value=$(echo "$entry" | jq -r '.to')

#      if [[ ! " ${from_values[@]} " =~ " $from_value " ]]; then
#           result="${result}${to_value},"
#           # Add this "from" value to the array
#           from_values+=("$from_value")
#      fi
# done

# result="${result%,}"

# rm gan_protein_structural_requirements/data/raw/model2/train_inhibitor_ids.txt
# echo "$result" > gan_protein_structural_requirements/data/raw/model2/train_inhibitor_ids.txt


# # turn ids into pdb ids
# train_enzyme_ids_data="$(cat gan_protein_structural_requirements/data/raw/model2/train_enzyme_ids.txt)"

# jobId="$(curl -s --form 'from="UniProtKB_AC-ID"' \
#      --form 'to="PDB"' \
#      --form "ids="$train_enzyme_ids_data"" \
#      https://rest.uniprot.org/idmapping/run | jq -r '.jobId')"

# read -p "Pause Time 1 second" -t 1

# json_data="$(curl -s "https://rest.uniprot.org/idmapping/stream/$jobId")"

# from_values=()

# # Extract values of "to" key and concatenate into a comma-separated string, skipping repeats
# result=""
# for entry in $(echo "$json_data" | jq -c '.results[]'); do
#      from_value=$(echo "$entry" | jq -r '.from')
#      to_value=$(echo "$entry" | jq -r '.to')

#      if [[ ! " ${from_values[@]} " =~ " $from_value " ]]; then
#           result="${result}${to_value},"
#           # Add this "from" value to the array
#           from_values+=("$from_value")
#      fi
# done

# result="${result%,}"

# rm gan_protein_structural_requirements/data/raw/model2/train_inhibitor_ids.txt
# echo "$result" > gan_protein_structural_requirements/data/raw/model2/train_inhibitor_ids.txt



# #################### test ##################

# # turn ids into pdb ids
# test_inhibitor_ids_data="$(cat gan_protein_structural_requirements/data/raw/model2/test_inhibitor_ids.txt)"

# jobId="$(curl -s --form 'from="UniProtKB_AC-ID"' \
#      --form 'to="PDB"' \
#      --form "ids="$test_inhibitor_ids_data"" \
#      https://rest.uniprot.org/idmapping/run | jq -r '.jobId')"

# read -p "Pause Time 1 second" -t 1

# json_data="$(curl -s "https://rest.uniprot.org/idmapping/stream/$jobId")"

# from_values=()

# # Extract values of "to" key and concatenate into a comma-separated string, skipping repeats
# result=""
# for entry in $(echo "$json_data" | jq -c '.results[]'); do
#      from_value=$(echo "$entry" | jq -r '.from')
#      to_value=$(echo "$entry" | jq -r '.to')

#      if [[ ! " ${from_values[@]} " =~ " $from_value " ]]; then
#           result="${result}${to_value},"
#           # Add this "from" value to the array
#           from_values+=("$from_value")
#      fi
# done

# result="${result%,}"

# rm gan_protein_structural_requirements/data/raw/model2/test_inhibitor_ids.txt
# echo "$result" > gan_protein_structural_requirements/data/raw/model2/test_inhibitor_ids.txt


# # turn ids into pdb ids
# test_enzyme_ids_data="$(cat gan_protein_structural_requirements/data/raw/model2/test_enzyme_ids.txt)"

# jobId="$(curl -s --form 'from="UniProtKB_AC-ID"' \
#      --form 'to="PDB"' \
#      --form "ids="$test_enzyme_ids_data"" \
#      https://rest.uniprot.org/idmapping/run | jq -r '.jobId')"

# read -p "Pause Time 1 second" -t 1

# json_data="$(curl -s "https://rest.uniprot.org/idmapping/stream/$jobId")"

# from_values=()

# # Extract values of "to" key and concatenate into a comma-separated string, skipping repeats
# result=""
# for entry in $(echo "$json_data" | jq -c '.results[]'); do
#      from_value=$(echo "$entry" | jq -r '.from')
#      to_value=$(echo "$entry" | jq -r '.to')

#      if [[ ! " ${from_values[@]} " =~ " $from_value " ]]; then
#           result="${result}${to_value},"
#           # Add this "from" value to the array
#           from_values+=("$from_value")
#      fi
# done

# result="${result%,}"

# rm gan_protein_structural_requirements/data/raw/model2/test_inhibitor_ids.txt
# echo "$result" > gan_protein_structural_requirements/data/raw/model2/test_inhibitor_ids.txt

# rm gan_protein_structural_requirements/data/raw/model2/uniprot_id_data.tsv
# echo "Extraction complete."