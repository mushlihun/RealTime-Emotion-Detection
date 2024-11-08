import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, classification_report
from keras.models import Sequential
from keras.layers import Dense

frozen_data = pd.read_excel('wajahsedihhasil.xlsx')
fresh_data = pd.read_excel('wajahgembirahasil.xlsx')

X_frozen = frozen_data.drop(columns=['Label'], errors='ignore')
X_fresh = fresh_data.drop(columns=['Label'], errors='ignore')

X = pd.concat([X_frozen, X_fresh], ignore_index=True)
y = ['ayam_beku'] * len(X_frozen) + ['wajah_gembira'] * len(X_fresh)
y_numeric = np.where(np.array(y) == 'wajah_sedih', 0, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y_numeric, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Output layer untuk klasifikasi biner

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_scaled, y_train, epochs=50, batch_size=10, validation_split=0.2)

y_probs = model.predict(X_test_scaled).flatten()

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

print("Threshold optimal:", optimal_threshold)

y_pred_custom_threshold = (y_probs >= optimal_threshold).astype(int)

print("Classification Report dengan Threshold Optimal:")
print(classification_report(y_test, y_pred_custom_threshold, target_names=["wajah_sedih", "wajah_gembira"]))

model.save('model.h5')  # Ganti dengan jalur yang diinginkan
print("Model saved successfully.")





#Pengujian Model
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

model = load_model('model.h5')

data1 = 22.6193650867028
data2 = 2811.86731617119
data3 = 53.0270432531477
data4 = 2.49481585963404
data5 = 2.04603890314984
data6 = 0.000000000000000000007684707601
data7 = 0.000000000000858890660633937000
data8 = 0.000000000000000000520465276533
data9 = 0.000000000599706163494537000000
data10 = 0.000000001256211175775330000000
data11 = 0.000002051535760999080000000000
data12 = 0.001897797637949880000000000000
data13 = 0.184618004821179000000000000000
data14 = 0.999734966246216000000000000000
data15 = 0.941595064284034000000000000000
data16 = 1.491040745741400000000000000000
data17 = 0.688538998749207000000000000000
data18 = 0.829782500869479000000000000000

data_sample = [
    [data1, data2, data3, data4, data5,
     data6, data7, data8,
     data9, data10, data11, 
     data12, data13, data14, data15, 
     data16, data17, data18]
]

single_data_df = pd.DataFrame(data_sample)

scaler = StandardScaler()
single_data_scaled = scaler.fit_transform(single_data_df)  # Ini hanya contoh, perlu pakai scaler yang sama dengan model training

prediction = (model.predict(single_data_scaled) > optimal_threshold).astype(int).flatten()

predicted_label = 'wajah_sedih' if prediction[0] == 0 else 'wajah_gembira'

print(f"Hasil prediksi: {predicted_label}")

