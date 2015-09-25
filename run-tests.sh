#!/bin/bash

nosetests-2.7 --with-coverage --cover-package deeplearning \
    --cover-xml \
    test integrationtest/testWorkflow.py
