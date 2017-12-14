from flask import Flask, request, Response
import json
import random
import os

ENVIRONMENT_NAME = "dev"
if 'ENVIRONMENT_NAME' in os.environ:
    ENVIRONMENT_NAME = os.environ['ENVIRONMENT_NAME']

app = Flask(__name__)


@app.route('/actual')
def actulRank():
    return Response(response=json.dumps("random"), status=200, mimetype="application/json")

@app.route('/rank/<rank_id>', methods=['POST'])
def rank(rank_id):
    article = request.json
    rank = random.random()
    return Response(response=json.dumps(rank), status=200, mimetype="application/json")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=(40409 if ENVIRONMENT_NAME == "dev" else 5000))
