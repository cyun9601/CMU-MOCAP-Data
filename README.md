Our main purpose is various processing of skeleton data. (i.e. Process data using only a few joints, normalize, Remove global translation, Missing marker)  
We worked on the asf/amc format of CMU mocap dataset, and we will going to add other data as well.


## CMU MOCAP Data

- We use the <a href='http://mocap.cs.cmu.edu/'>CMU Motion Capture Database</a> with 31 markers attached to actors, captured with an optical motion capture system. This data is converted from the joint angle representation in the original dataset to the 3D joint position.
- CMU Mocap dataset has several formats: ``asf/amc``, ``c3d``, ``vsk/v``, ``txt``. We represent these formats as skeleton data, represented by position values, and visualize in 3D catesian coordinate. 

- You can download the mocap data in here  
 - asf/amc: https://www.dropbox.com/sh/n9rlxn00az0fmnz/AABduEi-do8h-no-L7ozAbn4a?dl=1
 - ~~c3d: https://www.dropbox.com/sh/i2mxfi97o6of0rn/AABOSAJCaCSTukPpwTTJK_PYa?dl=1~~
 - ~~vsk/v:~~
 - ~~txt:~~


## Environment
- Python == 3.7.8
- numpy == 1.19.5
- transforms3d
- matplotlib


## Directory Structure

- AMCParser: To parse a asf and amc files. It is from https://github.com/CalciferZh/AMCParser.
- config: Configuration of main.py.
  - major
    - raw_data_name: Dataset of raw data.
    - raw_data_type: Data type of raw data.
    - num_joint: The number of joints of selected raw data.
    - node_mask: Node number to remove.
    - max_frame: frame size.
    - debug: Use only three raw data files when true.
- data: Directory of raw dataset. 
- processed_data: Directory of processed data. 
- main.py: Process and save data.
- visualization.ipynb: Visualization of processed data.


## Data Processing (main.py)

- This data is sub-sampled to 60 frames per second, and separated into overlapping windows of 64 frames (overlapped by half frames) which can be set in config file.
- The global translation is removed by subtracting the position of the root joint from the original data 
- Finally, we subtracted the mean pose from the original data and divided the absolute maximum value in each coordinate direction to normalize the data into [-1, 1].

```
python3 main.py './config/CMU/asfamc.yaml' 
```

## Visualization (visualization.ipynb)

-


## To do

- Add code about c3d, vsk/v, txt 
