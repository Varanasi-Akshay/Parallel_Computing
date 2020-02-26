#!/bin/bash

cat petsccg.out | awk '/Running/ {print} /KSPSolve/ {print $4}'
