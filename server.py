from flask import Flask, request

app = Flask(__name__)


@app.route('/sendMessage', methods=['POST'])
def send_message():
    # To send a curl to this endpoint do
    # curl -H "Content-Type: application/json" --request POST --data
    # '{"user_message": "from the user"}' http://localhost:5000/sendMessage
    request_json = request.get_json()
    user_message = request_json['user_message']
    print(user_message)
    # TODO pass message to retico
    # TODO return retico response
    # return json object
    return {
        'decision': 0,
        'message': 'from api'
    }
