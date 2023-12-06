#!/bin/bash

wget -O gan_protein_structural_requirements/data/raw/uniprot_id_data.tsv "https://rest.uniprot.org/uniprotkb/search?fields=accession%2Clength%2Cid&format=tsv&query=%28%28database%3ADrugBank%29%29+AND+%28reviewed%3Atrue%29+AND+%28proteins_with%3A1%29+AND+%28annotation_score%3A5%29+AND+%28length%3A%5B201+TO+400%5D%29&size=500&sort=accession+asc"


touch gan_protein_structural_requirements/data/raw/train_ids.txt
touch gan_protein_structural_requirements/data/raw/test_ids.txt

tsv_file="gan_protein_structural_requirements/data/raw/uniprot_id_data.tsv"
output_file_train="gan_protein_structural_requirements/data/raw/train_ids.txt"
output_file_test="gan_protein_structural_requirements/data/raw/test_ids.txt"
column_number=1

# Skip the first row (header), extract values, and split between two output files
tail -n +2 "$tsv_file" | cut -f"$column_number" | awk 'BEGIN {srand()} {if (rand() < 0.7) printf("%s,", $0) >> "'"$output_file_train"'"; else printf("%s,", $0) >> "'"$output_file_test"'"}'

# Remove trailing commas from both files
sed -i -e 's/.$//'  "$output_file_train"
sed -i -e 's/.$//' "$output_file_test"
rm gan_protein_structural_requirements/data/raw/train_ids.txt-e
rm gan_protein_structural_requirements/data/raw/test_ids.txt-e


#################### train ##################

# turn ids into pdb ids
train_ids_data="$(cat gan_protein_structural_requirements/data/raw/train_ids.txt)"

jobId="$(curl -s --form 'from="UniProtKB_AC-ID"' \
     --form 'to="PDB"' \
     --form "ids="$train_ids_data"" \
     https://rest.uniprot.org/idmapping/run | jq -r '.jobId')"

read -p "Pause Time 1 second" -t 1

json_data="$(curl -s "https://rest.uniprot.org/idmapping/stream/$jobId")"

result=""
for to_value in $(echo "$json_data" | jq -r '.results[] | .to'); do
     result="${result}${to_value},"
done

result="${result%,}"

rm gan_protein_structural_requirements/data/raw/train_ids.txt
echo "$result" > gan_protein_structural_requirements/data/raw/train_ids.txt


#################### test ##################

# turn ids into pdb ids
train_ids_data="$(cat gan_protein_structural_requirements/data/raw/test_ids.txt)"

jobId="$(curl -s --form 'from="UniProtKB_AC-ID"' \
     --form 'to="PDB"' \
     --form "ids="$train_ids_data"" \
     https://rest.uniprot.org/idmapping/run | jq -r '.jobId')"

read -p "Pause Time 1 second" -t 1

json_data="$(curl -s "https://rest.uniprot.org/idmapping/stream/$jobId")"

result=""
for to_value in $(echo "$json_data" | jq -r '.results[] | .to'); do
     result="${result}${to_value},"
done

result="${result%,}"


rm gan_protein_structural_requirements/data/raw/test_ids.txt
echo "$result" > gan_protein_structural_requirements/data/raw/test_ids.txt


rm gan_protein_structural_requirements/data/raw/uniprot_id_data.tsv
echo "Extraction complete. Results saved to $output_file_train and $output_file_test"