# Brain-Computer Interface : 2-Class Motor Imagery Classification using CSP/SVM
The script trains a classifier  on two different motor imagery datasets.

The first dataset is from BCI Competition IV data set 1 100Hz data which can be found here: https://www.bbci.de/competition/iv/, along with a description of the data. They have to be of format 'data/BCICIV_calib_ds1{letter}.mat' and be downloaded and placed in a folder named 'data' in the same directory as this file.

The files contain EEG recordings of users while doing two classes of motor imagery selected from the three classes left hand, right hand, and foot.
It contains methods that pre-process the EEG recording by applying bandpass filter as well as feature exctraction and selection using Common Spatial Patterns. Trains a Support Vector Machines model to classify between the two classes of movement. Can also evaluate the model and give an accuracy score, as well as provide figures that showcase the effects of epochings and preprocessing.

The second dataset is from the OpenBMI dataset which can be found here: http://dx.doi.org/10.5524/100542. They have to be of format 'sess01_subj0{num}_EEG_MI.mat' and be downloaded and placed in a folder named 'data' in the same directory as this file.

The files contain EEG recordings of users while doing motor imagery tasks of right and left hand movement. It also has methods that pre-process the EEG recording by applying bandpass filter as well as feature exctraction and selection using Common Spatial Patterns. Trains a Support Vector Machines model to classify between the two classes of movement. Can evaluate the model and give an accuracy score, as well as provide figures that showcase the effects of preprocessing.
