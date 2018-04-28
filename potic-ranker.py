from flask import Flask, request, Response
import json
import random
import os
import logging
import logging.config
from pymongo import MongoClient

from sklearn.base import BaseEstimator, TransformerMixin
import sklearn.preprocessing
import pickle
import numpy as np

ENVIRONMENT_NAME = "dev"
if 'ENVIRONMENT_NAME' in os.environ:
    ENVIRONMENT_NAME = os.environ['ENVIRONMENT_NAME']

MONGO_HOST = 'potic-mongodb'
MONGO_PORT = 27017
MONGO_DATABASE = 'potic'
MONGO_AUTH_DATABASE = 'admin'
MONGO_USERNAME = 'poticModels'
MONGO_PASSWORD = ''
if 'MONGO_PASSWORD' in os.environ:
    MONGO_PASSWORD = os.environ['MONGO_PASSWORD']

if ENVIRONMENT_NAME == 'dev':
    MONGO_HOST = '185.14.185.186'

mongo_client = MongoClient(host=MONGO_HOST, port=MONGO_PORT, username=MONGO_USERNAME, password=MONGO_PASSWORD, authSource=MONGO_AUTH_DATABASE)
potic_mongodb = mongo_client[MONGO_DATABASE]
models_mongodb = potic_mongodb.model

LOGZIO_TOKEN = None
if 'LOGZIO_TOKEN' in os.environ:
    LOGZIO_TOKEN = os.environ['LOGZIO_TOKEN']
if os.path.isfile('logzio-dev.properties'):
    with open('logzio-dev.properties', 'r') as logzioTokenFile:
        LOGZIO_TOKEN = logzioTokenFile.read().replace('\n', '')
if LOGZIO_TOKEN != None:
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
            }
        },
        'loggers': {
            'potic-ranker': {
                'handlers': ['console', 'logzio'],
                'level': 'INFO' if ENVIRONMENT_NAME == "prod" else 'DEBUG',
            }
        }
    }
else:
    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'verbose': {
                'format': '%(asctime)s [%(thread)d] %(levelname)s %(module)s - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO' if ENVIRONMENT_NAME == "prod" else 'DEBUG',
                'formatter': 'verbose'
            }
        },
        'loggers': {
            'potic-ranker': {
                'handlers': ['console'],
                'level': 'INFO' if ENVIRONMENT_NAME == "prod" else 'DEBUG',
            }
        }
    }

logging.config.dictConfig(LOGGING_CONFIG)


class CustomBinarizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        self.binarizer = sklearn.preprocessing.LabelBinarizer().fit(X["source"])
        return self

    def transform(self, X):
        source_tr = self.binarizer.transform(X["source"])
        return np.column_stack((X["word_count"], source_tr))


class CustomBinarizerNB(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        self.binarizer = sklearn.preprocessing.LabelBinarizer().fit(X["source"])
        return self

    def transform(self, X):
        source_tr = self.binarizer.transform(X["source"])
        return np.column_stack((X["word_count"], X["skipped_count"], X["showed_count"], source_tr))

random_model = { 'name': 'random', 'version': '1.0', 'description': 'random ranks' }
logreg_model = { 'name': 'logreg', 'version': '1.0', 'description': 'logistic regression (source, words count)' }
nbayes_model = { 'name': 'nbayes', 'version': '1.0', 'description': 'Bernoulli naive bayes (source, words count, showed count, skipped count)' }
svm_model = { 'name': 'svm', 'version': '1.0', 'description': 'svm (source, words count, showed count, skipped count)' }
all_models = [ random_model, logreg_model, nbayes_model, svm_model ]

app = Flask(__name__)


@app.route('/model', methods=['GET'])
def models():
    try:
        logging.getLogger('potic-ranker').debug("receive GET request for /model", extra={'loglevel':'DEBUG'})
        return Response(response=json.dumps(all_models), status=200, mimetype="application/json")
    except Exception as e:
        logging.getLogger('potic-ranker').error("GET request for /model failed: " + str(e), extra={'loglevel':'ERROR'})
        return Response(status=500)


@app.route('/model/<model_id>', methods=['POST'])
def model(model_id):
    try:
        logging.getLogger('potic-ranker').debug("receive POST request for /model/" + str(model_id) + "; body=" + str(request.json), extra={'loglevel':'DEBUG'})

        articles = request.json

        if model_id == logreg_model["name"] + ":" + logreg_model["version"]:
            if os.path.isfile('serialized_logreg'):
                with open('serialized_logreg', 'r') as serialized_model_file:
                    serialized_model_logreg = serialized_model_file.read()
                    return Response(response=json.dumps({ 'serialized_model': serialized_model_logreg }), status=200, mimetype="application/json")

        if model_id == nbayes_model["name"] + ":" + nbayes_model["version"]:
            if os.path.isfile('serialized_nbayes'):
                with open('serialized_nbayes', 'r') as serialized_model_file:
                    serialized_model_nbayes = serialized_model_file.read()
                    return Response(response=json.dumps({ 'serialized_model': serialized_model_nbayes }), status=200, mimetype="application/json")

        if model_id == svm_model["name"] + ":" + svm_model["version"]:
            if os.path.isfile('serialized_svm'):
                with open('serialized_svm', 'r') as serialized_model_file:
                    serialized_model_svm = serialized_model_file.read()
                    return Response(response=json.dumps({ 'serialized_model': serialized_model_svm }), status=200, mimetype="application/json")

        if model_id == random_model["name"] + ":" + random_model["version"]:
            return Response(response=json.dumps({ 'serialized_model': '' }), status=200, mimetype="application/json")

        logging.getLogger('potic-ranker').error("Unknown model " + str(model_id), extra={'loglevel':'ERROR'})
        return Response(status=404)
    except Exception as e:
        logging.getLogger('potic-ranker').error("POST request for /model/" + str(model_id) + " failed: " + str(e), extra={'loglevel':'ERROR'})
        return Response(status=500)


@app.route('/rank/<model_id>', methods=['POST'])
def rank(model_id):
    try:
        logging.getLogger('potic-ranker').debug("receive POST request for /rank/" + str(model_id) + "; body=" + str(request.json), extra={'loglevel':'DEBUG'})

        article = request.json

        source = article["source"] if "source" in article else ""
        word_count = int(article["word_count"]) if "word_count" in article else 0
        skipped_count = int(article["skipped_count"]) if "skipped_count" in article else 0
        showed_count = int(article["showed_count"]) if "showed_count" in article else 0

        if model_id == logreg_model["name"] + ":" + logreg_model["version"]:
            serialized_model_logreg = models_mongodb.find_one( { 'name': logreg_model["name"], 'version': logreg_model["version"] } ).serializedModel
            model_logreg = pickle.loads(serialized_model_logreg)

            model_input = np.array([(word_count, source)], dtype=[('word_count', 'int'), ('source', 'object')])
            rank = model_logreg.predict_proba(model_input)[0][1]
            logging.getLogger('potic-ranker').debug("calculated rank " + str(rank), extra={'loglevel': 'DEBUG'})
            return Response(response=json.dumps(rank), status=200, mimetype="application/json")

        if model_id == nbayes_model["name"] + ":" + nbayes_model["version"]:
            serialized_model_nbayes = models_mongodb.find_one( { 'name': nbayes_model["name"], 'version': nbayes_model["version"] } ).serializedModel
            model_nbayes = pickle.loads(serialized_model_nbayes)

            model_input = np.array([(word_count, skipped_count, showed_count, source)], dtype=[('word_count', 'int'), ('skipped_count', 'int'), ('showed_count', 'int'), ('source', 'object')])
            rank = model_nbayes.predict_proba(model_input)[0][1]
            logging.getLogger('potic-ranker').debug("calculated rank " + str(rank), extra={'loglevel': 'DEBUG'})
            return Response(response=json.dumps(rank), status=200, mimetype="application/json")

        if model_id == svm_model["name"] + ":" + svm_model["version"]:
            serialized_model_svm = models_mongodb.find_one( { 'name': svm_model["name"], 'version': svm_model["version"] } ).serializedModel
            model_svm = pickle.loads(serialized_model_svm)

            model_input = np.array([(word_count, skipped_count, showed_count, source)], dtype=[('word_count', 'int'), ('skipped_count', 'int'), ('showed_count', 'int'), ('source', 'object')])
            rank = model_svm.predict_proba(model_input)[0][1]
            logging.getLogger('potic-ranker').debug("calculated rank " + str(rank), extra={'loglevel': 'DEBUG'})
            return Response(response=json.dumps(rank), status=200, mimetype="application/json")

        if model_id == random_model["name"] + ":" + random_model["version"]:
            rank = random.random()
            return Response(response=json.dumps(rank), status=200, mimetype="application/json")

        logging.getLogger('potic-ranker').error("Unknown model " + str(model_id), extra={'loglevel':'ERROR'})
        return Response(status=404)
    except Exception as e:
        logging.getLogger('potic-ranker').error("POST request for /rank/" + str(model_id) + " failed: " + str(e), extra={'loglevel':'ERROR'})
        return Response(status=500)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=(40409 if ENVIRONMENT_NAME == "dev" else 5000))
