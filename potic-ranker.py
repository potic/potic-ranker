from flask import Flask, request, Response
import json
import random
import os
import logging
import logging.config

ENVIRONMENT_NAME = "dev"
if 'ENVIRONMENT_NAME' in os.environ:
    ENVIRONMENT_NAME = os.environ['ENVIRONMENT_NAME']

if 'LOGZIO_TOKEN' in os.environ:
    LOGZIO_TOKEN = os.environ['LOGZIO_TOKEN']
if os.path.isfile('logzio-dev.properties'):
    with open('logzio-dev.properties', 'r') as logzioTokenFile:
        LOGZIO_TOKEN = logzioTokenFile.read().replace('\n', '')
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '%(levelname)s %(asctime)s %(module)s %(process)d %(thread)d %(message)s'
        },
        'logzioFormat': {
            'format': '{"additional_field": "value"}'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'verbose'
        },
        'logzio': {
            'class': 'logzio.handler.LogzioHandler',
            'level': 'DEBUG',
            'formatter': 'logzioFormat',
            'token': LOGZIO_TOKEN,
            'logs_drain_timeout': 5,
            'url': 'https://listener.logz.io:8071',
            'debug': True
        },
    },
    'loggers': {
        'potic-ranker': {
            'handlers': ['console', 'logzio'],
            'level': 'DEBUG'
        }
    }
}
logging.config.dictConfig(LOGGING_CONFIG)

app = Flask(__name__)


@app.route('/actual')
def actulRank():
    try:
        logging.getLogger('potic-ranker').debug("receive GET request for /actual")
        return Response(response=json.dumps("random"), status=200, mimetype="application/json")
    except:
        logging.getLogger('potic-ranker').error("GET request for /actual failed")
        return Response(status=500)

@app.route('/rank/<rank_id>', methods=['POST'])
def rank(rank_id):
    try:
        logging.getLogger('potic-ranker').debug("receive POST request for /rank/" + rank_id)
        article = request.json
        rank = random.random()
        return Response(response=json.dumps(rank), status=200, mimetype="application/json")
    except:
        logging.getLogger('potic-ranker').error("POST request for /rank/" + rank_id + " failed")
        return Response(status=500)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=(40409 if ENVIRONMENT_NAME == "dev" else 5000))
