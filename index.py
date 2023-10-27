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
idMap={
1:100,2:101,3:103,4:104,5:105,6:106,7:107,8:108,9:109,10:111,11:112,12:113, 13:114, 14:115, 15:116, 16:117, 17:118, 18:119, 19:121, 20: 122, 21:123, 22:124, 23:200, 24:201, 25:202, 26:203, 27:205, 28:207, 29:208, 30:209, 31:210, 32:212, 33:213, 34:214 , 35:215, 36:217, 37:219, 38:220, 39:221, 40:223, 41:228, 42:230, 43:231 ,44:232, 45:233, 46:234, 47:102,
}
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
    countA=0
    countB=0

    for e in prediction:
        if e=='arrhythmia':
            countA=countA+1
        elif e=='N':
            countB=countB+1
    print("Count A:",countA," count B:",countB)
    text="{Total Instance : "+str(countA+countB)+" } "+" {Normal Instance :"+str(countB)+" } {Arrhythmia Instance : "+str(countA)+" }"
    label.config(text=text,font=("Arial", 16))
    print(prediction)
    for ax in axes:
        ax.clear()   
    # Plot each patient's signal
    axes[0].set_title("Pateint Id : "+str(selected_id) +  " Arrhythmia Detection")
    bars=axes[0].bar(['Normal','Arrhythmia'],[countB,countA])
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
id_combobox = ttk.Combobox(frame,font=("Arial", 14), values=list(range(1, 48)))
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
