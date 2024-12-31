from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pickle

# Load the trained model pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Initialize the Flask app
app = Flask(__name__)

# Define the list of teams and cities
teams = sorted([
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
])

cities = sorted([
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
])

# Store the result for display
result_data = {}

# Define the home route
@app.route("/")
def home():
    return render_template("index.html", teams=teams, cities=cities)

# Define the prediction route
@app.route("/predict", methods=["POST"])
def predict():
    global result_data

    # Retrieve form data
    batting_team = request.form["batting_team"]
    bowling_team = request.form["bowling_team"]
    city = request.form["city"]
    target = int(request.form["target"])
    score = int(request.form["score"])
    wickets = int(request.form["wickets"])
    overs = float(request.form["overs"])

    # Calculate derived features
    runs_left = target - score
    balls_left = 120 - int(overs * 6)
    remaining_wickets = 10 - wickets
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6 / balls_left) if balls_left > 0 else 0

    # Prepare the input DataFrame
    df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [remaining_wickets],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    # Get predictions from the model
    result = pipe.predict_proba(df)
    r_1 = round(result[0][0] * 100)  # Probability for bowling team
    r_2 = round(result[0][1] * 100)  # Probability for batting team

    # Store the result in a global variable
    result_data = {
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "batting_team_probability": r_2,
        "bowling_team_probability": r_1
    }

    # Redirect to the results page
    return redirect(url_for('result'))

# Define the result route
@app.route("/result")
def result():
    return render_template("result.html", result=result_data)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
