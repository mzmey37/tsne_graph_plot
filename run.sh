#!/bin/bash

if [ ! -d facebook ] # download files with graphs if they don't exist
then
    wget http://snap.stanford.edu/data/facebook.tar.gz
    tar -xvzf facebook.tar.gz
    rm facebook.tar.gz
fi

pip install -r requirements.txt

python run.py facebook/0.edges, facebook/107.edges, facebook/348.edges