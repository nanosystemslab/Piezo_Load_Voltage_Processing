# Piezo_Load_Voltage_Processing
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python Versions](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)](#)

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

## Features

- TODO

## Requirements

- TODO

## Installation
### Collect Repository
To clone the repository: 
```sh
git clone https://github.com/nanosystemslab/Small_Scale_Contact.git
cd Small_Scale_Contact
```

### Using pip
install dependencies using pip:
```sh
pip install .
```

### Using Poetry
Alternatively, if you prefer using Poetry for dependency management, you can run:
```sh
poetry install
```

## Using a Poetry Shell
To start a new shell session with your project's dependencies, use the following command:
```sh
poetry shell
```
This will activate a new shell where all the dependencies specified in your `pyproject.toml` are available.

## Usage
1. **Prepare Input Data**: Ensure your input data files are correctly formatted and placed in the `DATA` directory.
2. **Run Simulations**: Use the scripts in the `src` directory to generate surface data and plot models.
```sh
 python3 src/Piezo_Load_Voltage_Processing/plot_shidmazu.py -i DATA/C/shimadzu/60mm_min_250N--C-*csv
 python3 src/Piezo_Load_Voltage_Processing/plot_shidmazu.py -i DATA/C-100N/60mm_min_100N--C-*csv
```
3. **View Results**: Access the output files in the `OUT` directory and analyze the generated data.

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [GPL 3.0 license][license],
_Piezo_Load_Voltage_Processing_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from [@nanosystemslab]'s [Nanosystems Lab Python Cookiecutter] template.

[@nanosystemslab]: https://github.com/nanosystemslab
[pypi]: https://pypi.org/
[Nanosystems Lab Python Cookiecutter]: https://github.com/nanosystemslab/cookiecutter-nanosystemslab
[file an issue]: https://github.com/kailer-oko/Piezo_Load_Voltage_Processing/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/kailer-oko/Piezo_Load_Voltage_Processing/blob/main/LICENSE
[contributor guide]: https://github.com/kailer-oko/Piezo_Load_Voltage_Processing/blob/main/CONTRIBUTING.md
[command-line reference]: https://Piezo_Load_Voltage_Processing.readthedocs.io/en/latest/usage.html
