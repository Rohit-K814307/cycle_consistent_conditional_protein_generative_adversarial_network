#!/bin/sh

#display arguments
echo "id: $1"
echo "pdb_dir: $2"
echo "save_path: $3" #save_path is 'None' when no save

ID=$1
PDB_DIR=$2
SAVE_PATH=$2

python ./utils/protein_visualizer.py $ID $PDB_DIR $SAVE_PATH