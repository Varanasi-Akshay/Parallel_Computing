#!/bin/bash
gfortran -c smooth.f90 hw0.f90
gfortran hw0.f90 smooth.f90 -o hw0.out