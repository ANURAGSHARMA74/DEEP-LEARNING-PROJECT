# app.py
"""
Polished Toxic Comment Detector (ANN + TF-IDF) with modern Tkinter GUI.
Save as app.py and run: python app.py
Requires: pandas, numpy, nltk, scikit-learn, tensorflow, matplotlib, seaborn
"""
import os
import re
import string
import tempfile
import joblib
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn + nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ANN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Make sure NLTK resources exist
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# --------- Configuration ----------
DATA_PATH = r"D:\PROJECTS SEM-3MCA\DEEP LEARNING MINI PROJECT\test.csv"    # dataset file (must have comment_text,toxic)
MAX_FEATURES = 50000
NGRAM_RANGE = (1,2)
ANN_EPOCHS = 10
ANN_BATCH = 32
ANN_DROPOUT = 0.3
ANN_HIDDEN = 64
# ---------------------------------

# ------------------ Preprocessing ------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    t = str(text).lower()
    t = re.sub(r'http\S+|www\S+|https\S+', '', t)
    t = re.sub(r'<.*?>', '', t)
    t = re.sub(r'@\w+|#', '', t)
    t = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', t)
    t = re.sub(r'\d+', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    words = [lemmatizer.lemmatize(w) for w in t.split() if w not in stop_words]
    return ' '.join(words)

# ------------------ Load & Train Model ------------------
def build_and_train():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset '{DATA_PATH}' not found.")
    df = pd.read_csv(DATA_PATH)
    if 'comment_text' not in df.columns or 'toxic' not in df.columns:
        raise ValueError("CSV must have 'comment_text' and 'toxic' columns")
    df = df.dropna(subset=['comment_text']).reset_index(drop=True)
    df['clean_text'] = df['comment_text'].apply(clean_text)

    X = df['clean_text']
    y = df['toxic']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Vectorize
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, stop_words='english', ngram_range=NGRAM_RANGE, sublinear_tf=True)
    X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
    X_test_tfidf = vectorizer.transform(X_test).toarray()

    # Build ANN
    input_dim = X_train_tfidf.shape[1]
    model = Sequential([
        Dense(ANN_HIDDEN, input_shape=(input_dim,), activation='relu'),
        Dropout(ANN_DROPOUT),
        Dense(ANN_HIDDEN, activation='relu'),
        Dropout(ANN_DROPOUT),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train
    early = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_train_tfidf, 
            y_train,
            epochs=ANN_EPOCHS,
            batch_size=ANN_BATCH,
            validation_split=0.1,
            verbose=1,
            callbacks=[early])

    # Evaluate
    y_pred_prob = model.predict(X_test_tfidf).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    # Save artifacts
    model.save("ann_model.h5")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

    return {
        "model": model,
        "vectorizer": vectorizer,
        "accuracy": acc,
        "report": report,
        "cm": cm,
        "test_index": X_test.index
    }

# Build model
print("🔧 Training ANN model (may take a short while)...")
try:
    model_info = build_and_train()
except Exception as e:
    message = f"Error while preparing model: {e}"
    print(message)
    raise

model = model_info['model']
vectorizer = model_info['vectorizer']
MODEL_ACC = model_info['accuracy']
MODEL_REPORT = model_info['report']
MODEL_CM = model_info['cm']

print(f"ANN Model trained. Accuracy: {MODEL_ACC*100:.2f}%")
print("\nClassification Report:\n")
print(MODEL_REPORT)
print("\nConfusion Matrix:\n", MODEL_CM)

# ------------------ GUI ------------------
BG = "#f5f7fb"
CARD_BG = "#ffffff"
ACCENT = "#2b8cff"
TOXIC_COLOR = "#ff6b6b"
SAFE_COLOR = "#2ecc71"
TEXT = "#222222"

root = tk.Tk()
root.title("✨ Next-Gen Toxic Comment Detector (ANN)")
root.geometry("900x540")
root.configure(bg=BG)
root.resizable(False, False)

style = ttk.Style(root)
style.theme_use('clam')
style.configure("TFrame", background=BG)
style.configure("Card.TFrame", background=CARD_BG, relief="flat")
style.configure("Title.TLabel", font=("Helvetica", 18, "bold"), background=BG, foreground=TEXT)
style.configure("Header.TLabel", font=("Helvetica", 12, "bold"), background=CARD_BG, foreground=TEXT)
style.configure("Normal.TLabel", font=("Helvetica", 11), background=CARD_BG, foreground=TEXT)
style.configure("Accent.TButton", font=("Helvetica", 11, "bold"), foreground="white", background=ACCENT)
style.map("Accent.TButton", background=[('active', "#1c65ca")])

# Left frame
left = ttk.Frame(root, style="TFrame")
left.place(x=20, y=20, width=540, height=500)
# Right frame
right = ttk.Frame(root, style="TFrame")
right.place(x=580, y=20, width=300, height=500)

# Title & input
title = ttk.Label(left, text="Toxic Comment Detector", style="Title.TLabel")
title.pack(anchor="w", padx=12, pady=(6,8))
sub = ttk.Label(left, text="Type/paste a comment below (multi-line supported). Click 'Check' to predict.", background=BG, foreground="#555", font=("Helvetica", 10))
sub.pack(anchor="w", padx=12)
input_box = ScrolledText(left, wrap="word", font=("Helvetica", 12), width=60, height=9, padx=8, pady=8)
input_box.pack(padx=12, pady=10)

btn_frame = ttk.Frame(left, style="TFrame")
btn_frame.pack(padx=12, pady=(0,8), fill="x")

# Result card
card = ttk.Frame(left, style="Card.TFrame")
card.pack(padx=12, pady=10, fill="x")
result_label = ttk.Label(card, text="Result will appear here", style="Header.TLabel")
result_label.pack(anchor="w", padx=12, pady=(12,4))
prob_var = tk.DoubleVar(value=0.0)
prob_frame = ttk.Frame(card, style="Card.TFrame")
prob_frame.pack(fill="x", padx=12, pady=(0,10))
prob_bar_bg = tk.Canvas(prob_frame, height=18, bg=CARD_BG, highlightthickness=0)
prob_bar_bg.pack(fill="x")

# History
hist_label = ttk.Label(left, text="History (last checks):", font=("Helvetica", 11), background=BG)
hist_label.pack(anchor="w", padx=12)
history_listbox = tk.Listbox(left, height=6, font=("Helvetica", 10))
history_listbox.pack(padx=12, pady=(6,12), fill="x")

# ------------------ FUNCTIONS ------------------
def show_result(pred, proba, original_text):
    if pred == 1:
        color = TOXIC_COLOR
        text = "🚨 Toxic"
        detail = "This comment is predicted as TOXIC."
    else:
        color = SAFE_COLOR
        text = "✅ Not Toxic"
        detail = "This comment is predicted as NOT TOXIC."
    result_label.config(text=f"{text}  —  {detail}", background=CARD_BG, foreground=TEXT)
    # probability bar
    prob_bar_bg.delete("all")
    if proba is None:
        prob_bar_bg.create_rectangle(0,0,prob_bar_bg.winfo_width(),18, fill="#eee", outline="")
        prob_bar_bg.create_text(10,9, text="Probability not available", anchor="w", fill="#666", font=("Helvetica",9))
    else:
        w = prob_bar_bg.winfo_width()
        fill_w = int(w * proba)
        prob_bar_bg.create_rectangle(0,0,w,18, fill="#eee", outline="")
        prob_bar_bg.create_rectangle(0,0,fill_w,18, fill=color, outline="")
        prob_bar_bg.create_text(w-6,9, text=f"{proba*100:5.1f}%", anchor="e", fill="#fff" if proba>0.3 else "#333", font=("Helvetica",9,"bold"))
    # history
    short = original_text if len(original_text)<=80 else original_text[:77]+"..."
    history_listbox.insert(0, f"{text} — {short}")
    if history_listbox.size() > 10:
        history_listbox.delete(10, tk.END)

def do_check():
    text = input_box.get("1.0", "end").strip()
    if not text:
        messagebox.showwarning("Empty", "Please enter a comment to check.")
        return
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned]).toarray()
    pred_prob = model.predict(vec)[0][0]
    pred = int(pred_prob >= 0.5)
    show_result(pred, pred_prob, text)

def do_clear():
    input_box.delete("1.0", "end")

def copy_result():
    try:
        root.clipboard_clear()
        root.clipboard_append(result_label.cget("text"))
        messagebox.showinfo("Copied", "Result copied to clipboard.")
    except Exception as e:
        messagebox.showerror("Error", f"Could not copy: {e}")

# Buttons
check_btn = ttk.Button(btn_frame, text="Check Comment", style="Accent.TButton", command=do_check)
check_btn.grid(row=0, column=0, padx=(0,8))
clear_btn = ttk.Button(btn_frame, text="Clear", command=do_clear)
clear_btn.grid(row=0, column=1, padx=(0,8))
copy_btn = ttk.Button(btn_frame, text="Copy Result", command=copy_result)
copy_btn.grid(row=0, column=2, padx=(0,8))
exit_btn = ttk.Button(btn_frame, text="Exit", command=root.destroy)
exit_btn.grid(row=0, column=3)

# RIGHT PANEL: Model Info + Samples
r_card = ttk.Frame(right, style="Card.TFrame")
r_card.pack(padx=12, pady=12, fill="both", expand=True)
m_title = ttk.Label(r_card, text="Model Summary", style="Header.TLabel")
m_title.pack(anchor="w", padx=10, pady=(10,4))
acc_label = ttk.Label(r_card, text=f"Accuracy: {MODEL_ACC*100:.2f}%", style="Normal.TLabel")
acc_label.pack(anchor="w", padx=10)

# Samples
s_title = ttk.Label(r_card, text="Quick Samples", style="Header.TLabel")
s_title.pack(anchor="w", padx=10, pady=(8,4))
samples = [
    "Thank you for your help, much appreciated!",
    "You are an absolute idiot and a disgrace.",
    "I disagree respectfully with your point.",
    "Go die, you loser!",
    "This is a helpful comment, thanks!",
    "Shut up and stop talking nonsense."
]
def insert_sample(s):
    input_box.delete("1.0", "end")
    input_box.insert("1.0", s)
for s in samples:
    btn = ttk.Button(r_card, text=(s if len(s)<=36 else s[:33]+"..."), width=34, command=lambda st=s: insert_sample(st))
    btn.pack(padx=10, pady=4, anchor="w")

# Footer
footer = ttk.Label(root, text="Developed by Anurag", background=BG, foreground="#666", font=("Helvetica", 9))
footer.place(x=12, y=515)

# Canvas resize
def _update_canvas_width(event):
    prob_bar_bg.config(width=event.width)
prob_frame.bind("<Configure>", _update_canvas_width)

root.mainloop()