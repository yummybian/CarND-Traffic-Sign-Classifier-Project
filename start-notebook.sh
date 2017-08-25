#!/bin/bash

source activate carnd-term1
exec jupyter notebook &> /dev/null &
