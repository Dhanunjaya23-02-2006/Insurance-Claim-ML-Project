from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Get input data from the form
            input_features = [
                float(request.form["policy_type"]),
                float(request.form["accident_severity"]),
                float(request.form["driving_record"]),
            ]
            
            # Convert to numpy array and reshape for prediction
            input_array = np.array([input_features])
            prediction = model.predict(input_array)
            
            # Return result
            result = "Attorney Involved" if prediction[0] == 1 else "No Attorney Involved"
            return render_template("app.html", prediction=result)

        except Exception as e:
            return render_template("app.html", prediction=f"Error: {str(e)}")

    return render_template("app.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
