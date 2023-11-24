from flask import Flask, request
from flask_cors import CORS
from Runner import Runner
import pandas as pd

app = Flask(__name__)
CORS(app)

runner = Runner()

last_message = None

@app.route('/sendMessage', methods=['POST'])
def send_message():
    global last_message
    # To send a curl to this endpoint do
    # curl -H "Content-Type: application/json" --request POST --data
    # '{"user_message": "from the user"}' http://localhost:5000/sendMessage
    request_json = request.get_json()
    user_message = request_json
    print('user_message:', user_message)
    # pass correct decision to retico and wait for the response
    # TODO UNCOMMENT
    # decision = runner.get_dm_decision(user_message)
    last_message = user_message
    decision = 1
    print('decision', decision)

    # return json object
    return {
        'decision': decision,
    }


@app.route('/sendCorrectDecision', methods=['POST'])
def send_correct_decision():
    global last_message
    # To send a curl to this endpoint do
    # curl -H "Content-Type: application/json" --request POST --data
    # '{"user_message": "from the user"}' http://localhost:5000/sendMessage
    request_json = request.get_json()
    correct_decision = request_json
    print('correct decision:', correct_decision)
    # pass message to retico and wait for the response
    # TODO uncomment
    # response = runner.get_dm_response(correct_decision)
    data = {
        'Message': [last_message],
        'Correct_Decision': [correct_decision]
    }
    df = pd.DataFrame(data)
    df.to_csv('user_data.csv', mode='a', index=False, header=False)

    # return json object
    return {
        'status': 'ok',
    }
