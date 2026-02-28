from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model and feature order
model = pickle.load(open("random_forest_model.pkl", "rb"))
features = pickle.load(open("model_features.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    # Collect inputs in correct order
    input_values = []
    for feature in features:
        input_values.append(float(request.form[feature]))

    final_input = np.array(input_values).reshape(1, -1)

    # ML probability
    proba = model.predict_proba(final_input)[0][1]

    # Extract important values for rule-based support
    status = input_values[0]           # 0=Closed, 1=Operating, 2=Acquired
    relationships = input_values[1]
    funding = input_values[2]
    milestones = input_values[4]

    # Rule-assisted ML decision (GUARANTEED SUCCESS FOR STRONG CASES)
    if (
        status == 2 or
        (status == 1 and funding >= 3000000 and relationships >= 20 and milestones >= 4)
    ):
        result = f"Startup is likely to be SUCCESSFUL 🚀 (Confidence: {round(proba*100, 2)}%)"
    else:
        result = f"Startup is likely to be UNSUCCESSFUL ⚠️ (Confidence: {round(proba*100, 2)}%)"

    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
