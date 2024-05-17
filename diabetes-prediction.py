
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


Data = pd.read_csv('diabetes.csv')
x = Data.drop(columns='Outcome', axis=1)
y = Data['Outcome']


scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)


x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, stratify=y, random_state=2)


classifier = SVC(kernel='linear')
classifier.fit(x_train, y_train)

# Function to perform prediction and display result
def predict_diabetes():
    try:
        
        inputs = [float(entry.get()) for entry in entry_fields]
        
        new_data = np.array(inputs).reshape(1, -1)
        
        st_new_data = scaler.transform(new_data)
        
        prediction = classifier.predict(st_new_data)
        
        if prediction[0] == 0:
            messagebox.showinfo("Prediction", "The person is not diabetic")
        else:
            messagebox.showinfo("Prediction", "The person is diabetic")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numerical values.")

# Create GUI
root = tk.Tk()
root.title("Diabetes Prediction")


entry_fields = []
label_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPF", "Age"]

entry_fields = []
for i, label_name in enumerate(label_names):
    label = tk.Label(root, text=label_name + ":")
    label.grid(row=i, column=0, padx=10, pady=5, sticky="e")
    entry = tk.Entry(root)
    entry.grid(row=i, column=1, padx=10, pady=5, sticky="w")
    entry_fields.append(entry)


predict_button = tk.Button(root, text="Predict", command=predict_diabetes)
predict_button.grid(row=len(label_names), columnspan=2, padx=10, pady=10)
predict_button.grid(row=len(label_names)+1, columnspan=2, padx=10, pady=10, sticky="nsew")



root.mainloop()
7,196,90,0,0,39.8,0.451,41,1