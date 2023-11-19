#!/bin/sh

cd $PWD/gan_protein_structural_requirements/utils
mkdir esmfoldv1
cd esmfoldv1
wget -cO - https://huggingface.co/facebook/esmfold_v1/resolve/main/.gitattributes > .gitattributes
wget -cO - https://huggingface.co/facebook/esmfold_v1/resolve/main/config.json > config.json
wget -cO - https://huggingface.co/facebook/esmfold_v1/resolve/main/pytorch_model.bin > pytorch_model.bin
wget -cO - https://huggingface.co/facebook/esmfold_v1/resolve/main/special_tokens_map.json > special_tokens_map.json
wget -cO - https://huggingface.co/facebook/esmfold_v1/resolve/main/tokenizer_config.json > tokenizer_config.json
wget -cO - https://huggingface.co/facebook/esmfold_v1/resolve/main/vocab.txt > vocab.txt
echo "all downloads complete"