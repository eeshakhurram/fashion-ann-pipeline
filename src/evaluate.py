import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

d = np.load("data/processed/fashion_mnist.npz")
model = keras.models.load_model("models/model.h5")

loss, acc = model.evaluate(d["x_test"], d["y_test"], verbose=0)
pred = np.argmax(model.predict(d["x_test"]), axis=1)

os.makedirs("reports", exist_ok=True)
cm = confusion_matrix(d["y_test"], pred)
ConfusionMatrixDisplay(cm).plot()
plt.savefig("reports/confusion_matrix.png")

with open("metrics.json", "w") as f:
    json.dump({"test_loss": float(loss), "test_accuracy": float(acc)}, f, indent=2)

print("Test accuracy:", acc)