from flask import Flask, request
from Runner import Runner

app = Flask(__name__)

runner = Runner()

@app.route('/sendMessage', methods=['POST'])
def send_message():
    # To send a curl to this endpoint do
    # curl -H "Content-Type: application/json" --request POST --data
    # '{"user_message": "from the user"}' http://localhost:5000/sendMessage
    request_json = request.get_json()
    user_message = request_json['user_message']
    print('user_message:', user_message)
    # TODO pass message to retico
    # Process the user's message and wait for the response
    decision = runner.get_dm_decision(user_message)

    # return json object
    return {
        'decision': decision,
        'message': 'from api'
    }
