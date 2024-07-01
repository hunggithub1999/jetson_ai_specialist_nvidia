
from utils import *
from models import *
import sounddevice as sd
import numpy as np
import torch.nn.functional as F
import datetime
import time

# extract data from audio to spectrogram
def extract_audio_spec(signal):
    new_sr = 44100
    num_channel = 1
    maxMs =1000
    reaud = AudioUtil.resample(((signal,44100)), new_sr)
    rechan = AudioUtil.rechannel(reaud, num_channel)
    dur_aud = AudioUtil.pad_trunc(rechan, maxMs)
    sgram = AudioUtil.spectro_gram(dur_aud, n_mels=64, n_fft=1024, hop_len=None)
    return sgram

# record and detect audio
def detect_sound(fs, seconds):
    print("status of fan")
    t = str(datetime.datetime.now())
    my_record = sd.rec(int(seconds*fs), samplerate=fs, channels=1, device=[11,11])
    sd.wait()
    signal = torch.Tensor(my_record).t()
    data_spec = extract_audio_spec(signal)
    data_detect = np.array(data_spec).reshape(1,64,86)
    data_detect_mean = data_detect.mean()
    data_detect_std = data_detect.std()
    data_detect = (data_detect-data_detect_mean)/data_detect_std
    data_detect = data_detect.reshape(1,1,64,86)
    tensor = torch.tensor(data_detect)
    #print("tensor ", tensor)
    model = AudioClassifier()
    model.load_state_dict(torch.load("models/model_classificaiton_epoch_50.pt"))
    model.eval()
    output = model(tensor)
    _, prediction = torch.max(output,1)
    listLabelName = ["abnormal", "normal"]
    smax = F.softmax(output).detach().numpy()
    print("smax ",smax)
    if smax[0][0] > 0.8:
        print(t, " abnormal")
    else:
        print(t, " normal")


while True:
    try:
        print(" ")
        detect_sound(44100, 1)
        time.sleep(1)
        print("..")
    except KeyboardInterrupt:
        break
    except Exception as e:
        print("error ", e)
        continue



