from setuptools import setup, find_packages


setup(name='cycle_consistent_conditional_protein_generative_adversarial_network',
      version='1.0.0',
      description='Core code for Adversarially-Driven Generation of De Novo Proteins for Therapeutic Drug Design.',
      license="MIT",
      author='Rohit Kulkarni',
      author_email='rkulka@gmu.edu',
      packages=find_packages(),
      install_requires=['numpy', 'matplotlib'])