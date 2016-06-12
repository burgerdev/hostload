#!/bin/bash

nosetests-2.7 --with-coverage --cover-package tsdl \
    --cover-xml \
    test integrationtest
