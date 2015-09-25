#!/bin/bash

nosetests-2.7 --with-coverage --cover-package deeplearning \
    --cover-html \
    test integrationtest/testWorkflow.py
