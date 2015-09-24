#!/bin/bash

nosetests --with-coverage --cover-package deeplearning \
    test integrationtest/testWorkflow.py
