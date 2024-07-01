from models import *
from utils import *
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
from torch.utils.data import random_split
import os
import numpy as np
torch.manual_seed(2)

# ----------------------------
# Sound Dataset
# ----------------------------
class SoundDS(Dataset):
  def __init__(self, df, data_path):
    self.df = df
    self.data_path = str(data_path)
    self.duration = 1000
    self.sr = 44100
    self.channel = 1
            
  # ----------------------------
  # Number of items in dataset
  # ----------------------------
  def __len__(self):
    return len(self.df)    
    
  # ----------------------------
  # Get i'th item in dataset
  # ----------------------------
  def __getitem__(self, idx):
    # Absolute file path of the audio file - concatenate the audio directory with
    # the relative path
    audio_file = self.data_path + self.df.loc[idx, 'relative_path']
    # Get the Class ID
    class_id = self.df.loc[idx, 'classID']

    aud = AudioUtil.open(audio_file)
    # Some sounds have a higher sample rate, or fewer channels compared to the
    # majority. So make all sounds have the same number of channels and same 
    # sample rate. Unless the sample rate is the same, the pad_trunc will still
    # result in arrays of different lengths, even though the sound duration is
    # the same.
    reaud = AudioUtil.resample(aud, self.sr)
    rechan = AudioUtil.rechannel(reaud, self.channel)
    dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
    sgram = AudioUtil.spectro_gram(dur_aud, n_mels=64, n_fft=1024, hop_len=None)

    return sgram, class_id



# Get file path
def pre_audio_filepath(data_path):
    df = pd.DataFrame()
    df_classID = []
    df_relative_path = []
    
    list_class = sorted(os.listdir(data_path))
    for index_class, name_class in enumerate(list_class):
        list_file_path = os.listdir(data_path+"/"+str(name_class))
        # print(listFilePath)
        for file_path in list_file_path:
            df_classID.append(index_class)
            df_relative_path.append(str(name_class)+"/"+file_path)
            
    df["classID"]=df_classID
    df["relative_path"]=df_relative_path
    return df

dir_data_train = "data/data_train/"
df = pre_audio_filepath(dir_data_train)

myds = SoundDS(df, dir_data_train)

# Random split of 80:20 between training and validation
num_items = len(myds)
num_train = round(num_items * 1.0)
num_val = num_items - num_train
train_ds, val_ds = random_split(myds, [num_train, num_val])

# Create training and validation data loaders
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)

# Create the model and put it on the GPU if available
myModel = AudioClassifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)
# Check that it is on Cuda
next(myModel.parameters()).device

# ----------------------------
# Training Loop
# ----------------------------
def training(model, train_dl, num_epochs):
    # Loss Function, Optimizer and Scheduler
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=int(len(train_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')

    # Repeat for each epoch
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        # Repeat for each batch in the training set
        for i, data in enumerate(train_dl):
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs,1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            #if i % 10 == 0:    # print every 10 mini-batches
            #    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
        
        # Print stats at the end of the epoch
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction/total_prediction
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')
        if epoch%10==0:
            torch.save(model.state_dict(),f"models/model_classificaiton_epoch_50_epoch_{epoch}.pt")
    torch.save(model.state_dict(),"models/model_classificaiton_epoch_50.pt")
    print('Finished Training')
  



# convert data form audio to spectrogram
def convert_wav_to_spec(audio_file, sr, channel, duration):
    aud = AudioUtil.open(audio_file)
    # Some sounds have a higher sample rate, or fewer channels compared to the
    # majority. So make all sounds have the same number of channels and same 
    # sample rate. Unless the sample rate is the same, the pad_trunc will still
    # result in arrays of different lengths, even though the sound duration is
    # the same.
    reaud = AudioUtil.resample(aud, sr)
    rechan = AudioUtil.rechannel(reaud, channel)
    dur_aud = AudioUtil.pad_trunc(rechan, duration)
    sgram = AudioUtil.spectro_gram(dur_aud, n_mels=64, n_fft=1024, hop_len=None)
    return sgram

# evaluate model 
def evaluateModel(dirModel, dirData, mode):
    listFileName = os.listdir(dirData)
    model_inf = AudioClassifier()
    model_inf.load_state_dict(torch.load(dirModel))
    model_inf.eval()
    dem = 0
    total_file = 0
    wrong_file = 0
    for fileName in (listFileName):
        try:
            spec = convert_wav_to_spec(dirData+fileName, 44100, 1, 1000)
            data = np.array(spec)
            data_mean = data.mean()
            data_std = data.std()
            data = (data - data_mean)/data_std
            data = data.reshape(1, 1, 64, 86)
            tensor = torch.tensor(data)
            output = model_inf(tensor)
            _, prediction = torch.max(output,1)
            if (int(prediction[0]))==mode:
                dem+=1
            else:
                wrong_file+=1
                pass
            total_file+=1
        except Exception as e: 
            # print(e)
            continue
    print(dirData, np.round(dem/total_file,2))
    return np.round(dem/total_file,2)


num_epochs=50 
#training(myModel, train_dl, num_epochs)

# data train
#evaluateModel("models/model_classificaiton_epoch_50.pt","data/data_train/normal/", 1)
#evaluateModel("models/model_classificaiton_epoch_50.pt","data/data_train/abnormal/", 0)
print("----------------------")
# data test
#evaluateModel("models/model_classificaiton_epoch_50.pt","data/data_test/normal/", 1)
#evaluateModel("models/model_classificaiton_epoch_50.pt","data/data_test/abnormal/", 0)
evaluateModel("models/model_classificaiton_epoch_50.pt","data/temp/", 0)

print("the end ...")
