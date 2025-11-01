# ===============================================
# EMAIL SPAM DETECTION USING LSTM + TKINTER GUI
# ===============================================

import numpy as np
import pandas as pd
import re
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam

# ===============================================
# STEP 1. LOAD DATASET
# ===============================================
df = pd.read_csv(r"D:\Ashish\DL\Project\cleaned_email_dataset.csv")

# Rename columns if needed (just to be sure)
df.columns = ['Email Text', 'Label']

# ===============================================
# STEP 2. CLEAN TEXT
# ===============================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['Cleaned'] = df['Email Text'].apply(clean_text)

# ===============================================
# STEP 3. TOKENIZE AND PAD
# ===============================================
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['Cleaned'])

X = tokenizer.texts_to_sequences(df['Cleaned'])
max_len = 200
X_padded = pad_sequences(X, maxlen=max_len, padding='post', truncating='post')

y = df['Label'].values

# ===============================================
# STEP 4. SPLIT DATA
# ===============================================
X_train, X_test, y_train, y_test = train_test_split(
    X_padded, y, test_size=0.2, random_state=42
)

# ===============================================
# STEP 5. BUILD LSTM MODEL
# ===============================================
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=max_len),
    Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# ===============================================
# STEP 6. TRAIN MODEL
# ===============================================
print("\nTraining the model, please wait...\n")
model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_test, y_test),
    verbose=1
)

# ===============================================
# STEP 7. EVALUATE MODEL
# ===============================================
loss, acc = model.evaluate(X_test, y_test)
print(f"\n Model trained successfully! Test Accuracy: {acc:.4f}\n")

# ===============================================
# STEP 8. PREDICTION FUNCTION FOR GUI
# ===============================================
def predict_email_gui():
    email_text = text_box.get("1.0", "end-1c").strip()
    if not email_text:
        messagebox.showwarning("Input Error", "Please enter some text!")
        return

    cleaned = clean_text(email_text)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    pred = float(model.predict(pad)[0][0])

    if pred > 0.5:
        result_label.config(text=f" Spam Email Detected! ({pred*100:.2f}%)", fg="red")
    else:
        result_label.config(text=f" Not a Spam Email ({(1-pred)*100:.2f}%)", fg="green")

# ===============================================
# STEP 9. TKINTER GUI
# ===============================================
root = tk.Tk()
root.title("Email Spam Detector (LSTM)")
root.geometry("650x450")
root.config(bg="#f8f9fa")

tk.Label(root, text=" Email Spam Detection", font=("Arial", 18, "bold"), bg="#f8f9fa").pack(pady=10)
tk.Label(root, text="Type or paste an email below:", font=("Arial", 12), bg="#f8f9fa").pack()

text_box = tk.Text(root, height=10, width=70, wrap="word", font=("Arial", 10))
text_box.pack(pady=10)

tk.Button(root, text="Check Spam", font=("Arial", 12, "bold"), bg="#007bff", fg="white",
          command=predict_email_gui).pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 14, "bold"), bg="#f8f9fa")
result_label.pack(pady=10)

root.mainloop()
