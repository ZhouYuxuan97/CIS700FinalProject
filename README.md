# CIS700FinalProject

This code was run using the following version/libraries:

## Python version:

Python 3.9.12

## Libraries:

Cuda v11.3.0, dill v0.3.4, matplotlib v3.5.2, numpy    v.1.21.5, TensorFlow v2.6.0, and PyTorch v1.11.0

## Instruction

main.py requires data_utils.py and cnn_1d_lstm.py to run. It also requires the holstep dataset, as available for download at http://cl-informatik.uibk.ac.at/cek/holstep/

local folders named "models" and "loss" have been filled with my trained result. 

Run plot.py can directly generate the same plot I showed in the final report. If you want to retrain the model, please remove all files under "models" and "loss" folders and run main.py. 

Notice:  main.py may take a long time to generate the vocab file, I suggest download this file from my Google Drive [link](https://drive.google.com/file/d/1Ru3EmKixqoTip3BEpxJqqTid4_K-tpoU/view?usp=sharing) and place it to the root folder(same with main.py) to save time.

