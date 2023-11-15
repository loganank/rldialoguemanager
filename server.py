from flask import Flask, request
from flask_cors import CORS
from Runner import Runner

app = Flask(__name__)
CORS(app)

runner = Runner()


@app.route('/sendMessage', methods=['POST'])
def send_message():
    # To send a curl to this endpoint do
    # curl -H "Content-Type: application/json" --request POST --data
    # '{"user_message": "from the user"}' http://localhost:5000/sendMessage
    request_json = request.get_json()
    user_message = request_json
    print('user_message:', user_message)
    # pass correct decision to retico and wait for the response
    decision = runner.get_dm_decision(user_message)

    # return json object
    return {
        'decision': decision,
    }


@app.route('/sendCorrectDecision', methods=['POST'])
def send_correct_decision():
    # To send a curl to this endpoint do
    # curl -H "Content-Type: application/json" --request POST --data
    # '{"user_message": "from the user"}' http://localhost:5000/sendMessage
    request_json = request.get_json()
    correct_decision = request_json
    print('correct decision:', correct_decision)
    # TODO pass message to retico and wait for the response
    response = runner.get_dm_response(correct_decision)

    # return json object
    return {
        'status': 'ok',
    }
