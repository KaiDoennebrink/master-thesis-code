# Deep active learning for ECG classification

## Installation

Create a virtual environment with conda, for example, install all requirements from `requirements.txt`, and activate the virtual environment.

## Command line interface (CLI)

The repository offers a CLI for the most important tasks. 
Run the commands from the base directory of the repository.
To get an overview about all commands run the command

```bash
main.py --help
```

To get the help for a specific command, e.g., prepare-data-ptbxl, run the following in your command line.

```bash
main.py prepare-data-ptbxl --help
```

## Datasets

The following datasets are used in this repository:
- PTB-XL (version: 1.0.3) from [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/)
- TPB-XL+ (version: 1.0.1) from [PhysioNet](https://physionet.org/content/ptb-xl-plus/1.0.1/)

## Troubleshooting

If you run into problems with OpenCV in a WSL environment, ensure that you install OpenCV to the Linux system directly: `sudo apt-get install python3-opencv`.
