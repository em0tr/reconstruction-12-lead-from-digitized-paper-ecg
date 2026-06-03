# Reconstruction of 12-lead from digitized paper electrocardiogram

Electrocardiograms (ECGs) have traditionally been stored as paper printouts, and this practice is still
common in many hospitals. However, when the original high-resolution waveform data is
transferred to paper, the quality is progressively reduced through several steps:
1. The 10-second recording is compressed to just 2.5 seconds per lead in order to display all 12 leads
on a single page.
2. The signal is plotted and printed on paper, which significantly lowers its resolution.
3. The paper ECG is then scanned and stored in the electronic patient record, a process that can
introduce additional artifacts such as rotation, skew, and distortion.

In previous research it has been shown that it is possible to restore the 2.5 seconds 12-lead digital ECG
from paper ECG [[Stenhede 2024]](https://www.cinc.org/archives/2024/pdf/CinC2024-262.pdf) and the aim of this project is to reconstruct the 10-second signal
from the 2.5 second ECG, which will enhance the usability of historical ECGs.

## Install
1. Clone the repository:
    ```shell
    git clone https://github.com/em0tr/reconstruction-12-lead-from-digitized-paper-ecg.git
    ```
2. Create a virtual environment and install required packages:
    ```shell
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
3. Create an input directory for the input dataset and put the dataset in that directory:
    ```shell
    mkdir input
    ```
   
## Running the program
The main.py file has three main modes: processing ECGs, getting statistics from processed ECGs and plotting processed ECGs. 
When processing ECGs data is saved using SQLite so that it is easy to do analysis or plotting. It is also possible to save the data to a CSV file, but if there is a lot of data CSV may be very slow to use. Using SQLite also saves the ECG signal for each ECG and lead, but since this massively increases the file size, it is not available for saving to CSV files. The ECG signal is only used for plotting the original and reconstructed signals overlaid.  
The file [util.py](util.py) contains multiple variables that may need to be changed for your data, such as the number of leads used, the number of ECGs and which leads were used. 

### Processing ECGs
Running the processing pipeline for all ECGs and leads:
```shell
python --process_ecg
```

To save the processed data using SQLite include:
```shell
python --process_ecg --save_db
```

To only do a subset of the ECGs or leads, use:
```shell
python --process_ecg --ecg_end=100 --lead_end=3
```
This will only process the first 100 ECGs and the first three leads.

To use a different delineate method and processing method use:
```shell
python --process_ecg --process_method=neurokit --delineate_method=dwt
```
Only the $\texttt{dwt}$ and $\texttt{cwt}$ methods are supported for delineating, since the other options result in different output dataframes that don't feature some of the necessary columns that are needed for this project. 

### Visualizing ECGs
**This requires the ECGs to have been processed**.

Running the visualizing pipeline for all ECGs and leads:
```shell
python --visualize_ecg
```

To save the plots include:
```shell
python --visualize_ecg --save_plots
```

### Statistics
**This requires the ECGs to have been processed**.

Running the statistics pipeline:
```shell
python --stats
```

### Splitting the data to individual CSV files for leads
**This requires the ECGs to have been processed**.

To split the data to individual CSV files for a CSV file for each lead use:
```shell
python split_data
```
