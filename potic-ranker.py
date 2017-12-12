from flask import Flask, request, Response
import json
import random

app = Flask(__name__)


@app.route('/rank', methods=['POST'])
def rank():
    articles = request.json
    ranks = map(lambda x: random.random(), articles)
    return Response(response=json.dumps(ranks), status=200, mimetype="application/json")


if __name__ == '__main__':
    app.run(host='0.0.0.0')
