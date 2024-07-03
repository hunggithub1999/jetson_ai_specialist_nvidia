# jetson_ai_specialist_nvidia

# The project is detect abnormal device by audio data analysis

##	Overview
The device that i use to apply model detection is my fan. In big factory, warehouse, ... People often use high-capacity fans to cool devices that operate continuously. When the fan is running, something may get stuck in the fan blades, which can lead to damage to the cooling fan device. By analyzing the sound of the fan when running normally and simulating an abnormal error (inserting a small piece of paper into the fan blade), the solution can recognize the fan's status. From there, send a warning to the user when the fan is in an abnormal state, helping to reduce damage and extend the fan's lifespan.
This solution can be applied in the field of industrial plants, helping to solve abnormal problems of equipment based on audio signal analysis.

##	Technical
process audio data 
build model deeplearning use pytorch

##	Hardware
DLAP-211-Nano - NVIDIA
Lcd mini 7 inch
micro record audio

#-----------------------------------------------------------

# How to run project
## Step 1: download source code
Download source code with path "https://github.com/hunggithub1999/jetson_ai_specialist_nvidia.git"
The source code will include data and script to build and run model realtime

## Step 2: Install all dependencies
install new virtual environment python for coding and running
install dependency in file requirement.txt

## Step 3: Setup data and train model
In folder data, I have include data train and data test with sound of the fan when it run normal and when it have issue.
You can use your own data if you want.
Run file train_model.py to train model sound detection with 50 epoch

## step 4: Run script detect abnormal sound realtime
Run file detect_realtime.py to detect status of fan in realtime
There are two stage of fan, one is fan run normal , one that I simulated the error by inserting a piece of paper while the fan was spinning