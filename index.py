path = './mitbih_database/'
window_size = 180
maximum_counting = 10000

classes = ['N', 'L', 'R', 'A', 'V']
n_classes = len(classes)
count_classes = [0]*n_classes

X = list()
y = list()

import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os
import pickle

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import csv

import pywt
from scipy import stats
model_file_path = './model.pkl'  
# %matplotlib inline

#IMPORT ARITHMIA CSV
df1= pd.read_csv('MIT-BIH Arrhythmia Database.csv')

#tranformation
x_data = df1.iloc[:, 2:]
y_label = df1[['type']]
X_train, X_test, y_train, y_test = train_test_split(x_data, y_label,test_size=0.4, random_state=101)

min_max_scaler = preprocessing.MinMaxScaler()
X_train_scaled = min_max_scaler.fit_transform(X_train)
print(X_train_scaled)

#MODEL IMPORT
with open(model_file_path, 'rb') as file:
    loaded_model = pickle.load(file)

# id to data maping
idMap={1:101, 2:106, 3: 108, 4:109, 5:112, 6:114, 7:115, 8:116, 9:118, 10:119, 11:122, 12:124,13: 201, 14:203, 15:205, 16:207,17: 208, 18:209,
 19:215, 20:220, 21:223, 22:230, 23:100, 24:103, 25:105, 26:111, 27:113, 28:117, 29:121, 30:123, 31:200, 32:202, 33:210, 34:212,35: 213, 36:214,
 37:219, 38:221,39: 222, 40:228, 41:231, 42:232, 43:233, 44:234}
#MAKE PREDICTION
def DetectA():
    selected_id = int(id_combobox.get())
    id =idMap[selected_id]
    print("id : ",id)
    dfnew=df1[df1['record']==id]
    x_data = dfnew.iloc[:, 2:]
    X_data_scaled = min_max_scaler.transform(x_data)
    # print(x_data)
    prediction = loaded_model.predict(X_data_scaled)
    cntA=0
    cntN=0
    cntSVEB=0
    cntVEB=0
    cntF=0
    cntQ=0
    print(prediction)
    for e in prediction:
        if e=='N':
            cntN=cntN+1
        elif e=='VEB':
            cntVEB=cntVEB+1
        elif e=='SVEB':
            cntSVEB=cntSVEB+1
        elif e=='F':
            cntF=cntF+1
        elif e=='Q':
            cntQ=cntQ+1
    print("Count A:",cntA," count B:",cntN,"cntVEB :",cntVEB,", cntSVEB",cntSVEB," cntF:",cntF,", cntQ:",cntQ)
    text="{ Total Instance : " + str(len(prediction)) + " } " + " { Normal Instance :" + str(cntN) + " } { Arrhythmia Instance : " + str(cntVEB+cntSVEB+cntF+cntQ)+" => | VEB : "+str(cntVEB)+" | SVEB : "+str(cntSVEB)+" | F : "+str(cntF)+" | Q : "+str(cntQ)+"| }"
    label.config(text=text,font=("Arial", 16))
    print(prediction)
    for ax in axes:
        ax.clear()   
    # Plot each patient's signal
    axes[0].set_title("Pateint Id : "+str(selected_id) +  " Arrhythmia Detection")
    bars=axes[0].bar(['Normal','VEB','SVEB','F','Q'],[cntN,cntVEB,cntSVEB,cntF,cntQ])
    for bar in bars:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

# Create a Tkinter canvas

    canvas.draw()

    # DetectA(dfnew)



plt.rcParams["figure.figsize"] = (30,6)
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['lines.color'] = 'b'
plt.rcParams['axes.grid'] = True 

def denoise(data): 
    w = pywt.Wavelet('sym4')
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    threshold = 0.04 # Threshold for filtering

    coeffs = pywt.wavedec(data, 'sym4', level=maxlev)
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))
        
    datarec = pywt.waverec(coeffs, 'sym4')
    
    return datarec

# Read files
filenames = next(os.walk(path))[2]
# print("filenames : ", filenames)
# Split and save .csv , .txt 
records = list()
annotations = list()
filenames.sort()


# segrefating filenames and annotations
for f in filenames:
    filename, file_extension = os.path.splitext(f)
    
    # *.csv
    if(file_extension == '.csv'):
        records.append(path + filename + file_extension)

    # *.txt
    else:
        annotations.append(path + filename + file_extension)



# GUI CODE
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Records
def visualizeECG(id_no):

    for r in range(id_no,id_no+1):
        signals = []
        for ax in axes:
            ax.clear()

        with open(records[id_no], 'rt') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|') # read CSV file\
            row_index = -1
            for row in spamreader:
                if(row_index >= 0):
                    signals.insert(row_index, int(row[1]))
                row_index += 1
        
        
        if r is id_no:
            print(len(signals))
            # Plot each patient's signal
            axes[0].set_title("pateint id : "+str(id_no) +  " Wave")
            axes[0].plot(signals[0:700])
           
            

        signals = denoise(signals)
        # Plot an example to the signals
        if r is id_no:
            # Plot each patient's signal
            axes[1].set_title("pateint id : "+str(id_no) +" wave after denoised")
            axes[1].plot(signals[0:700])
           
            
        #Normalization
        signals = stats.zscore(signals)
     
        if r is id_no:
            axes[2].set_title("pateint id : "+str(id_no) + " wave after z-score normalization ")
            axes[2].plot(signals[0:700])
            
        
        canvas.draw()


def display_ecg_signal():
    selected_id = int(id_combobox.get())

#     has_arrhythmia = arrhythmia_data.get(selected_id, "Data not available")
#     result_label.config(text=f"ID {selected_id} has Arrhythmia: {has_arrhythmia}")

    # You should replace this with your own ECG signal visualization logic
    # Sample ECG signal plot
    # dfnew=df1[df1['record']==101]
    # x_data = dfnew.iloc[:, 2:]
    # print(x_data)
    # # DetectA(dfnew)
    visualizeECG(selected_id)

# Create the main window
root = tk.Tk()

root.title("Arrhythmia Detector")

frame = tk.Frame(root)
frame.pack(pady=15)

# Create a label
title_label = tk.Label(frame, text="Select an ID:",font=("Arial", 14))
title_label.pack(side="left",padx=5)

# Create a ComboBox to select the ID
id_combobox = ttk.Combobox(frame,font=("Arial", 14), values=list(range(1, 44)))
id_combobox.pack(side="left",padx=10)

# Create a button to display the ECG signal
display_button = tk.Button(frame, text="Display ECG", font=("Arial", 14),command=display_ecg_signal)
display_button.pack(side="left",padx=10)

display_button2 = tk.Button(frame, text="Detect",font=("Arial", 14), command=DetectA)
display_button2.pack(side="left",padx=10)

label = tk.Label(root, text="")
label.pack(pady=20)

# Create a label to display the result
fig, axes = plt.subplots(3, figsize=(20, 10))
plt.subplots_adjust(hspace=0.5)
axes = axes.ravel()  # Flatten the 2D array of axes

# Create a canvas to display the plots
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

root.mainloop()
