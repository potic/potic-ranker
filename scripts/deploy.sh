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

warn "Pulling latest docker image..."
docker pull potic/potic-ranker:$TAG_TO_DEPLOY

warn "Currently running docker images"
docker ps -a

warn "Killing currently running docker image..."
docker kill potic-ranker; docker rm potic-ranker

warn "Starting docker image..."
docker run -dit --name potic-ranker --restart on-failure --link potic-mongodb -e ENVIRONMENT_NAME=$ENVIRONMENT_NAME -e MONGO_PASSWORD=$MONGO_PASSWORD -e LOGZIO_TOKEN=$LOGZIO_TOKEN potic/potic-ranker:$TAG_TO_DEPLOY

warn "Wait 30sec to check status"
sleep 30

warn "Currently running docker images"
docker ps -a
