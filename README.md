## MISP2021 Task 1 - Wake word spottig (WWS) baseline systems

- **Introduction**

    This is the system open source code of the manuscript entitled "Audio-Visual Wake Word Spotting in MISP2021 Challenge: Dataset Release and Deep Analysis". In the paper, we describe and release publicly the audio-visual wake word spotting (WWS) database in the MISP2021 Challenge. The task concerns the identification of predefined wake word(s) in utterances. ‘1’ indicates that the sample contains wake word, and ‘0’ indicates the opposite. For more information, please refer to the MISP Challenge task 1 description.


- **System description**

    The audio system implements a neural network (NN) based approach, where filter bank features are first extracted for each sample, and a network consisting of CNN layers, LSTM layer and fully connected layers are trained to assign labels to the audio samples.

    For the video system, the same network structure is adopted as the audio network. 
    
    For fusion, we consider late fusion. Specifically, the output posterior probabilities from audio-only and video-only systems are weighted and selected to calculate the final audio-visual final score.

- **preparation**

  - **prepare data directory**

      For training, development, and evaluation sets, we prepare data directories by extracting the downloaded zip compressed file to the current folder.

      ```
      unzip  -d ./  *.zip

      *.zip indicates the file name that needs to be unzipped
      ```

  - **speech simulation** 

    Simulating reverberant and noisy data from near field speech, noise is widely adopted. We provide a baseline speech simulation tool to add reverberation and noise for speech augmentation. Considering that the negative samples are easier to obtain, we simulate all positive samples and partial negative samples (listed in file [data_prepare/negative_simulation.scp](data_prepare/negative_simulation.scp)). Here, we only use channel 1 for simulation.

    - **add reverberation**

        An open-source toolkit [pyroomacoustic](https://github.com/LCAV/pyroomacoustics) is used to add reverberation. The room impulsive response (RIR) is generated according to the actual room size and microphone position.

    - **add noise**

        We provide a simple tool to add noise with different signal-to-noise ratio. In our configuration, the reverberated speech is corrupted by the collected noise at seven signal-to-noise ratios (from -15dB to 15dB with a step of 5dB).

    The pretrained model needs to be placed on the spectific path
    
    ```
    ${task1_baseline}/kws_net_only_video/model/pretrained/lipreading_LRW.pt
    ```
  - **download database**      

    ```    
    https://challenge.xfyun.cn/misp_dataset
    ```

     
- **Audio Wake Word Spotting**

    For features extraction, we employ 40-dimensional filter bank (FBank) features normalized by global mean and variance as the input of the audio WWS system. The final output of the models compared with the preset threshold after sigmoid operation to calculate the false reject rate (FRR) and false alarm rate (FAR).

- **Video Wake Word Spotting**

    To get visual embeddings, we firstly crop mouth ROIs from video streams, then use the [lipreading TCN](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks)  to extract 512-dimensional features. The extracted features are input into the same network structure as the audio network.


## Setting Paths

- **data prepare**

```
# Here, the given tool can simulate the positive samples directly. If you need to simulate the negative samples, you need to modify the default configuration.
--- data_prepare/run.sh ---
# Defining corpus directory
data_root=
# Defining path to python interpreter
python_path=
```

- **kws_net_only_audio**

```
--- kws_net_only_audio/run.sh ---
# Defining corpus directory
data_root=
# Defining path to python interpreter
python_path=
```

- **kws_net_only_video**

```
--- kws_net_only_video/run.sh ---
# Defining corpus directory
data_root=
# Defining path to python interpreter
python_path=
```

## Running the baseline audio system

- **Simulation (optional)**

    ```
    cd data_prepare
    sh run.sh
    ```

- **Run Audio Training**

    ```
    cd ../kws_net_only_audio
    sh run.sh
    ```

- **Run Video Training**

    ```
    cd ../kws_net_only_vudio
    sh run.sh
    ```

- **Run Fusion**

    ```
    cd ../kws_net_fusion
    python fusion.py
    ```

## Requirments

- **pytorch**

- **python packages:**

    numpy
    
    [OpenCV](https://github.com/opencv/opencv-python)

    tqdm

    [pyroomacoustic](https://github.com/LCAV/pyroomacoustics)

    [soundfile](https://github.com/bastibe/python-soundfile)

- **other tools:**

    [sox](http://sox.sourceforge.net/) 

