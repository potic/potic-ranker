#!/usr/bin/env sh

##############################################################################
##
##  Stop and kill currently running docker image, pull newest version and
##  run it.
##
##############################################################################

warn ( ) {
    echo "$*"
}

warn "Currently running docker images"
docker ps -a

warn "Killing currently running docker image..."
docker kill potic-ranker; docker rm potic-ranker

warn "Pulling latest docker image..."
docker pull potic/potic-ranker:$TAG_TO_DEPLOY

warn "Starting docker image..."
docker run -dit --name potic-ranker -e ENVIRONMENT_NAME=$ENVIRONMENT_NAME -e LOGZIO_TOKEN=$LOGZIO_TOKEN potic/potic-ranker:$TAG_TO_DEPLOY

warn "Currently running docker images"
docker ps -a
