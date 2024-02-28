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

Different datasets are used in this repository.

### PTB-XL (version: 1.0.3) 
from [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/)

Is used in the active learning cycles as input and the labels are used as expert labels, i.e., expensive but accurate.

### PTB-XL+ (version: 1.0.1) 
from [PhysioNet](https://physionet.org/content/ptb-xl-plus/1.0.1/)

Diagnostic statements from the 12SL algorithm are used in the active learning cycle as low-cost labels that are error-prune. 
Furthermore, the conversions from 12SL labels and PTB-XL labels to SNOMED codes are used.

### ICBEB 2018
from [Challenge website](http://2018.icbeb.org/Challenge.html)

Is used to pre-train a classifier that can be used in different settings.

### ECG Arrhythmia (version: 1.0.0) 
from [PhysioNet](https://physionet.org/content/ecg-arrhythmia/1.0.0/)

Is used to pre-train a classifier that can be used in different settings.

| Number of ECGs | ECG channels | Number of patients | ECG duration | Sample frequency | Number of classes                           |
|----------------|--------------|--------------------|--------------|------------------|---------------------------------------------|
| 45152          | 12           | 45152              | 10 seconds   | 500Hz            | 11 rhythms + 67 additional cardiac findings |


## Troubleshooting

If you run into problems with OpenCV in a WSL environment, ensure that you install OpenCV to the Linux system directly: `sudo apt-get install python3-opencv`.
