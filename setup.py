from setuptools import setup, find_packages

# Function to read the requirements file and return a list of dependencies
def read_requirements():
    with open('requirements.txt', 'r', encoding='utf-8') as req_file:
        requirements = req_file.read().splitlines()
    # Filter out editable installs
    requirements = [req for req in requirements if not req.startswith('-e')]
    return requirements

# Reading long description with specific encoding
with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='deforestation_free_finance',
    version='0.1.0',
    author='Marc Bohnet, Malte Hessenius, Tycho Tax',
    author_email='marc@climcom.org, malte@climcom.org, tycho@climcom.org',
    description='Making deforestation due diligence work in practice',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ClimateAndCompany/deforestation_free_finance',
    packages=find_packages(),
    install_requires=read_requirements(),
    python_requires='>=3.6',
)