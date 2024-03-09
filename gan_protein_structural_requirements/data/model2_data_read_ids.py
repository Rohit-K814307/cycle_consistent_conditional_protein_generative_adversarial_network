import pandas as pd
import os
import requests
import time


def map_one_id(id):
    response = requests.post(
        "https://rest.uniprot.org/idmapping/run",
        data={
            "from": "UniProtKB_AC-ID",
            "to": "PDB",
            "ids": id,
        },
    )

    response.raise_for_status()

    job_id = response.json()["jobId"]
    time.sleep(1)

    # Get mapping results
    results_url = f"https://rest.uniprot.org/idmapping/stream/{job_id}"
    response = requests.get(results_url)
    response.raise_for_status()

    json_data = response.json()

    if len(json_data["results"]) > 0:
        return json_data["results"]
    else:
        return False
    

def make_unique(dict_list):
    new_dict = {}

    for entry in dict_list:
        new_dict[entry["from"]] = entry["to"]

    return new_dict


def sift_ids(result_dict_inhibit, initial_ids_inhibit, result_dict_enzyme, initial_ids_enzyme):
    out_pairs = []
    
    inhibit_enzyme_pair = {key: value for key in initial_ids_inhibit for value in initial_ids_enzyme}

    inhib = make_unique(result_dict_inhibit)
    enzyme = make_unique(result_dict_enzyme)

    for key1 in inhib.keys():
        for key2 in enzyme.keys():
            if key1 in inhibit_enzyme_pair.keys():
                if inhibit_enzyme_pair[key1] == key2:
                    out_pairs.append([inhib[key1], enzyme[key2]])


    return out_pairs


def map_enzyme_inhibit_ids(id_enzymes, id_inhibits):
    mapped_inhibit = map_one_id(id_inhibits)
    mapped_enzyme = map_one_id(id_enzymes)

    if mapped_inhibit is not False and mapped_enzyme is not False:
        return sift_ids(mapped_inhibit, id_inhibits, mapped_enzyme, id_enzymes)
    else:
        return False
        


min_len = 0
max_len = 500

df = pd.read_table(f"{os.getcwd()}/gan_protein_structural_requirements/data/raw/model2/uniprot_id_data.tsv")


train_ids_enzyme = []
train_ids_inhibitor = []
test_ids_enzyme = []
test_ids_inhibitor = []

inhibitors = [] 
enzymes = []

for i in range(len(df["Entry"])):

    if df["Length"][i] <= max_len and pd.isna(df["Interacts with"][i]) == False:

        inhibitors.append(df["Entry"][i])

        if ";" in df["Interacts with"][i]:
            ind = df["Interacts with"][i].index(";")

            enzymes.append(df["Interacts with"][i][0:ind])
        else:
            enzymes.append(df["Interacts with"][i])

len_train = int(0.8 * len(inhibitors))

for i in range(len_train):
    train_ids_enzyme.append(enzymes[i])
    train_ids_inhibitor.append(inhibitors[i])
for i in range(len_train, len(inhibitors)):
    test_ids_enzyme.append(enzymes[i])
    test_ids_inhibitor.append(inhibitors[i])

train_mapped_enzyme = []
train_mapped_inhibitor = [] 

combined = map_enzyme_inhibit_ids(train_ids_enzyme, train_ids_inhibitor)
for i in range(len(combined)):
    train_mapped_enzyme.append(combined[i][1])
    train_mapped_inhibitor.append(combined[i][0])

test_mapped_enzyme = []
test_mapped_inhibitor = []

combined = map_enzyme_inhibit_ids(test_ids_enzyme, test_ids_inhibitor)
for i in range(len(combined)):
    test_mapped_enzyme.append(combined[i][1])
    test_mapped_inhibitor.append(combined[i][0])







f = open(f"{os.getcwd()}/gan_protein_structural_requirements/data/raw/model2/train_inhibitor_ids.txt", "w")
f.write(",".join(train_mapped_inhibitor))
f.close()

f = open(f"{os.getcwd()}/gan_protein_structural_requirements/data/raw/model2/train_enzyme_ids.txt", "w")
f.write(",".join(train_mapped_enzyme))
f.close()

f = open(f"{os.getcwd()}/gan_protein_structural_requirements/data/raw/model2/test_inhibitor_ids.txt", "w")
f.write(",".join(test_mapped_inhibitor))
f.close()

f = open(f"{os.getcwd()}/gan_protein_structural_requirements/data/raw/model2/test_enzyme_ids.txt", "w")
f.write(",".join(test_mapped_enzyme))
f.close()