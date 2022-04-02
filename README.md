## MISP2021 Task 1 - Wake word spottig (WWS) baseline systems

- **Introduction**

    This is the baseline system for the Multimodal Information Based Speech Processing Challenge 2021 (MISP2021) challenge task 1. This task concerns the identification of predefined wake word(s) in utterances. ‘1’ indicates that the sample contains wake word, and ‘0’ indicates the opposite. For a more detailed description see MISP Challenge task description.


- **System description**

    The audio system implements a neural network (NN) based approach, where filter bank features are first extracted for each sample, and a network consisting of CNN layers, LSTM layer and fully connected layers are trained to assign labels to the audio samples.

    For the video system, the same network structure is adopted as the audio network. Only middle field video is used to train the video system.
    
    For fusion, we consider late fusion, that is, the mean of the a posteriori probability of the network output of audio and video is used to calculate the final score.

- **preparation**

  - **prepare data directory**

      For training, development, and test sets, we prepare data directories by extracting the downloaded zip compressed file to the current folder.

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

  - **download the pretrained model**

    ```
    resources_link waiting…
    ```

    The pretrained model needs to be placed on the spectific path
    
    ```
    ${task1_baseline}/kws_net_only_video/model/pretrained/lipreading_LRW.pt
    ```
  - **download coordinate information used to crop ROI**
      - **midfield**
      
    ```
    resources_link waiting…
    ```
      - **far-field**         

    ```    
    resources_link waiting…
    ```

     
- **Audio Wake Word Spotting**

    For features extraction, we employ 40-dimensional filter bank (FBank) features normalized by global mean and variance as the input of the audio WWS system. The final output of the models compared with the preset threshold after sigmoid operation to calculate the false reject rate (FRR) and false alarm rate (FAR).

- **Video Wake Word Spotting**

    To get visual embeddings, we firstly crop mouth ROIs from video streams, then use the [lipreading TCN](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks)  to extract 512-dimensional features. The extracted features are input into the same network structure as the audio network.

## Results

  After adjusting different thresholds, the system performances are as follows:

| Modality       | Scenario    | FRR     | FAR    | Score  |
| -------------- | ----------- | ------- | -------|-------|
| Audio   | Middle      |  0.07   | 0.08   | 0.15   |
| Audio   | Far         |  0.18   | 0.09   |  0.27| 
| Video   | Middle      | 0.12|0.21 |0.33 | 
| Audio-visual | Middle      |  0.07 | 0.06 | 0.13 |
| Audio-visual | Far         |  0.12 | 0.14 | 0.26 |


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

