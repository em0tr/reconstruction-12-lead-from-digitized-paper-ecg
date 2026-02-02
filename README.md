# Reconstruction of 12-lead from digitized paper electrocardiogram

Electrocardiograms (ECGs) have traditionally been stored as paper printouts, and this practice is still common in many hospitals. However, when the original high-resolution waveform data is transferred to paper, the quality is progressively reduced through several steps:
1. The 10-second recording is compressed to just 2.5 seconds per lead in order to display all 12 leads on a single page.
2. The signal is plotted and printed on paper, which significantly lowers its resolution.
3. The paper ECG is then scanned and stored in the electronic patient record, a process that can introduce additional artifacts such as rotation, skew, and distortion.

In previous research we have shown that it is possible to restore the 2.5 seconds 12-lead digital ECG from paper ECG ([Stenhede 2024](https://www.cinc.org/archives/2024/pdf/CinC2024-262.pdf)) and the aim of this project is to reconstruct the 10 second signal from the 2.5 second ECG, which will enhance the usability of historical ECGs. 
