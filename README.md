# Sound Event Localization and Detection
This repo includes python scripts for: 
1. SELDnet highlighted in the [manuscript](https://arxiv.org/abs/1807.00129) and the original code can be cloned through this [repo](https://github.com/sharathadavanne/seld-dcase2019)
2. RawSELDnet is a modification over the original SELDnet to use the raw data isntead of spectrograms.
3. Generating simulated data using Pyroomacoustics library to test the performance of SELDnet and other SLEDnet based networks.
4. DOAnet highlighted in this [manuscript](https://arxiv.org/abs/1710.10059).
5. Beamformer for the purpose of audio classification and DOA estimation.

It includes also Android Studio projects for:
1. Alarm detection.
2. ESC50 classification.

## DCASE 2019 Task3 Dataset:
The description of both the task and dataset can be found on [the challenge website](http://dcase.community/challenge2019/task-sound-event-localization-and-detection#audio-dataset) and [this manuscript](https://arxiv.org/abs/1905.08546).


## SELDNet:
This network is the baseline of the DCASE 2019 challenge task3 and it basically takes both globaly normalized phase and amplitude of the spectrogram in order to feed a multitask learning network that predicts both class and direction of arrival for the active sound events.

The first part of the network taht predicts the class, ends with a sigmoid activated fully connected layer with 11 outputs (one for each class). The second part of the network ends with a linearly activated fully connected layer with 22 outputs (11 for azimuth and 11 for elevation). The labels of DOA are fed to the network in radians and they are all scaled to be between -180 and 180 degrees(although elevation angles were between -50 and 50 degrees, they were rescaled to be consistent with the azimuth angles).

Although this implementation uses the same network architecture as the original paper, it deploys a different data generator (to load batches on the fly) and invokes sacred for logging the run measurements.

To generate the spectrograms and labels used to train and test this network, use the script called [batch_gen_spec_feat.py](./batch_gen_spec_feat.py). The training is done through the script called [seld.py](./seld.py). You just need to change the paths to dataset and features in each of those two scripts (under names datadir and labelsdir). You can also control the run parameters like the window and hop length (for calculating spectrograms) through changing the corresponding parameters in [data_preparation.py](./OrigSELDNet/data_preparation.py) and [seld.py](./seld.py).

## RawSELDNet
This network is based on multi-task learning as well to predict both direction of arrival and sound events in multichannel audio data; however, it uses the raw signals as input instead of spectrograms' amplitude and phase in case of the baseline. Other than this, the network shares the same prediction architecture for both DOA and SE with the baseline network mentioned previously.

To split the raw data into windows and generate the corresponding labels used to train and test this network, use the script called [batch_gen_raw_feat.py](./batch_gen_raw_feat.py). The training is done through the script called [rawseld.py](./rawseld.py). you just need to change the paths to dataset and features in each of those two scripts (under names datadir and labelsdir). The run parameters including the window size, hop length, batch size, etc. are also in [data_preparation.py](./RawSELDNet/data_preparation.py) and [rawseld.py](./rawseld.py).

## Pyroomacoustics simulated data
The objective of this was to generate a simulated dataset similar to the DCASE19 task 3 dataset but with simpler configuration. The same set of isolated sound events (11 classes) which were used for generating the challenge dataset, were used as well to generate this dataset. The data for the 11 isolated classes can be downloaded from an older DCASE challenge [DCASE16_task2](http://www.cs.tut.fi/sgn/arg/dcase2016/task-sound-event-detection-in-synthetic-audio#audio-dataset). 

The class [roomIR](./RIR_cls.py) includes the room configuration including the number of microphones, isolated sound classes, the different directions of arrival, sampling frequency for input and output data, number of records to be genrated, each record length, and room shape and dimensions. You can run this [script](./gen_RR.py) to generate the room response.

You can run the baseline code with this simulated data using the corresponding modified scripts in this [directory](./SimSELDNet) and here is the [main](./main.py) to run.

## Adaptive Beamforming for Multi-channel Audio Data
This implementation is based on the practices implemented in the following two manuscripts: 
1. [Deep Long Short-Term Memory Adaptive Beamforming Networks For Multichannel Robust Speech Recognition](https://arxiv.org/abs/1711.08016).
2. [Neural Network Adaptive Beamforming for Robust Multichannel Speech Recognition](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45399.pdf).

The implementation here is divided into three main parts generating the beamformed audio data for the DCASE19 Task3 dataset through using the [BeamformIt](https://github.com/xanguera/BeamformIt.git) toolbox (the instructions for compiling the toolbox are in the attached GitHub repo). The second part is the LSTM-based model inetnded for performing the beamforming and generating filters' coefficients for the filter and sum stage. The third part includes the model that uses the beamformed signals produced by the beamforming model to classify and detect the sound events.



## Android Application for Alarm Detection
This includes the Android Studio project for the Android application of the alarm detection. The application core resides in this [directory](./android/app/src/main/java/org/bosch/alarm/) which includes the java classes for the app. The [MainActivity](./android/app/src/main/java/org/bosch/alarm/MainActivity.java) class performs basically the recording through creating a circular buffer and continuously filling it with data from microphone in a separate thread to ensure independence from the main prediction pipeline. Another thread is created in order to perform the prediction by loading the tesnrflow model and the required API's (like NNAPI or including GPU delegate if the model arch is supported by tensorflow-lite-gpu which currently supports only 2D CNN) and accessing the recording buffer through the global shared pointers that indicate the start and end of the segment inside the circular buffer that is continuously fed by new data from microphone. The prediction probability vector is produced and further processed/displayed through the main thread itself. You can find the main prameters controlling the run of the application in this class as well, including the window size, sampling frequency, and paths to tensorflow lite model file and class labels file (can be found [here](./android/app/src/main/assets)).

The [RecognizeAlarms](./android/app/src/main/java/org/bosch/alarm/RecognizeAlarms.java) class is dedicated to smoothing the prediction results overtime and comparing the current window prediction results with the previous ones. The class takes the probability vector and produces another result class that indicate if the dominating class for the current window is new or just like the last couple of predictions. You can always run the application without this class, as it is not mandatory for producing the probability vector of the classes which is produced actually in the main class mentioned above.

The main layouts for the app GUI can be found [here](./android/app/src/main/res/layout) and it includes the configuration of the bar chart displayed text fields and other view parameters.


## ESC50 Android Application
This Android studio project has the same structure as the previously menttion app for alarm detection; however, it doesn't use the smoothing class. An extra add to this project is a class that includes methods to sort the prediction vector and get the top K classes (probabilities and labels) which will be injected into the bar chart API for displaying. This class can be found [here](./ESC50/app/src/main/java/org/bosch/ESC50/Utilities.java) and it basically uses a hash map to store the probability/label pairs and then a tree map that is known to sort while insertion to sort the pairs in a descending order. The first K elements are then popped out to populate the bar chart.

