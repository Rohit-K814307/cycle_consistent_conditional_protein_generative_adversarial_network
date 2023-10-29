from setuptools import setup, find_packages


setup(name='gan_protein_structural_requirements',
      version='1.0.0',
      description='Core code for Adversarially-Driven Generation of De Novo Proteins Given Structural Requirements.',
      license="MIT",
      author='Rohit Kulkarni',
      author_email='rkulka@gmu.edu',
      packages=find_packages(),
      install_requires=['numpy', 'matplotlib'])