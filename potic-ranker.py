from flask import Flask, request, Response
import json
import random
import os
import logging
import logging.config

from sklearn.base import BaseEstimator, TransformerMixin
import sklearn.preprocessing
import pickle
import numpy as np

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
            'format': '%(asctime)s [%(thread)d] %(levelname)s %(module)s - %(message)s'
        },
        'logzioFormat': {
            'format': '{"env": "' + ENVIRONMENT_NAME + '", "service": "potic-ranker"}'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO' if ENVIRONMENT_NAME == "prod" else 'DEBUG',
            'formatter': 'verbose'
        },
        'logzio': {
            'class': 'logzio.handler.LogzioHandler',
            'level': 'INFO' if ENVIRONMENT_NAME == "prod" else 'DEBUG',
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
            'level': 'INFO' if ENVIRONMENT_NAME == "prod" else 'DEBUG',
        }
    }
}
logging.config.dictConfig(LOGGING_CONFIG)

app = Flask(__name__)


class CustomBinarizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        source_tr = sklearn.preprocessing.LabelBinarizer().fit(X["source"]).transform(X["source"])
        return np.column_stack((X["word_count"], source_tr))

model = None
if os.path.isfile('serialized_model'):
    with open('serialized_model', 'r') as serialized_model_file:
        model = pickle.loads(serialized_model_file.read())


@app.route('/actual')
def actulRank():
    try:
        logging.getLogger('potic-ranker').debug("receive GET request for /actual", extra={'loglevel':'DEBUG'})
        return Response(response=json.dumps("logreg"), status=200, mimetype="application/json")
    except:
        logging.getLogger('potic-ranker').error("GET request for /actual failed", extra={'loglevel':'ERROR'})
        return Response(status=500)

@app.route('/rank/<rank_id>', methods=['POST'])
def rank(rank_id):
    try:
        logging.getLogger('potic-ranker').debug("receive POST request for /rank/" + rank_id, extra={'loglevel':'DEBUG'})
        article = request.json
        source = article["card"]["source"]
        word_count = int(article["fromPocket"]["word_count"])
        model_input = np.array([(word_count, source)], dtype=[('word_count', 'int'), ('source', 'object')])
        rank = model.predict_proba(model_input)[0][1]
        logging.getLogger('potic-ranker').debug("received word_count " + word_count + ", source " + source + ", calculated rank " + rank,
                                                extra={'loglevel': 'DEBUG'})
        #rank = random.random()
        return Response(response=json.dumps(rank), status=200, mimetype="application/json")
    except Exception as e:
        logging.getLogger('potic-ranker').error("POST request for /rank/" + rank_id + " failed: " + e, extra={'loglevel':'ERROR'})
        return Response(status=500)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=(40409 if ENVIRONMENT_NAME == "dev" else 5000))
