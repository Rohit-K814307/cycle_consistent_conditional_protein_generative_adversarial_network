# Target: install-dependencies
install-dependencies:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Dependencies installed successfully."

# Target: run-setup-script
gan-protein-structural-requirements-data-raw-batch-2-data:
	./gan_protein_structural_requirements/scripts/data_download.sh -f ./gan_protein_structural_requirements/scripts/list_file.txt -o ./gan_protein_structural_requirements/data/raw/batch_2_data -p

# Target: project-setup
project-setup: install-dependencies run-setup-script
	@echo "Project setup completed."
