#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect all 12 feature inputs from the form
        features = [
            float(request.form["feature1"]),
            float(request.form["feature2"]),
            float(request.form["feature3"]),
            float(request.form["feature4"]),
            float(request.form["feature5"]),
            float(request.form["feature6"]),
            float(request.form["feature7"]),
            float(request.form["feature8"]),
            float(request.form["feature9"]),
            float(request.form["feature10"]),
            float(request.form["feature11"]),
            float(request.form["feature12"]),
        ]

        # Convert the list to a numpy array
        features = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)

        # Interpret result
        result = "Liver Disease Detected" if prediction[0] == 1 else "No Liver Disease"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)

    


# In[ ]:




