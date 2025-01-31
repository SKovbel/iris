main: help

SHELL := /bin/bash

BASE_PATH = $(shell pwd)
export CONTAINER = gzros2-humble
export SIM_PATH = $(BASE_PATH)/sim
export SRC_PATH = $(BASE_PATH)/../src

data:
	C:/work/python39/python.exe -m dataset

stats:
	C:/work/python39/python.exe -m stats

torch-nn-softmax:
	C:/work/python39/python.exe -m xtorch.nn_softmax

torch-gen-vae:
	C:/work/python39/python.exe -m xtorch.gen_vae

mse_sigmoid:
	C:/work/python39/python.exe -m xsimple.nn_mse_sigmoid

mle_softmax:
	C:/work/python39/python.exe -m xsimple.nn_mle_softmax

bayes_montecarlo_likelihood:
	C:/work/python39/python.exe -m xsimple.bayes_montecarlo_likelihood

help:
	@echo "# Available commands:"
	@echo ""
	@echo "make test0         # "
	@echo ""
