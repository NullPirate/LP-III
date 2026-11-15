import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv("Churn_Modelling.csv")
print(df.head())

X = df.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Exited'])
y = df['Exited']

X = pd.get_dummies(X, columns=['Geography', 'Gender'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential([
     Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
     Dropout(0.2),
     Dense(16, activation='relu'),
     Dense(1, activation='sigmoid')
])

model.compile(
     optimizer='adam',
     loss='binary_crossentropy',
     metrics=['accuracy']
)

early_stop = EarlyStopping(
     monitor='val_loss', patience=5, restore_best_weights=True
)

model.fit(
     X_train, y_train,
     epochs=50,
     batch_size=32,
     validation_split=0.2,
     callbacks=[early_stop],
     verbose=1
)

y_prob = model.predict(X_test)
y_pred = (y_prob >= 0.5).astype(int).ravel()

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("0 = Customer stays, 1 = Customer leaves")
print("The neural network predicts whether a customer will leave the bank.")
