# Writing ToDo list

基本的な話は発火率モデルですべてすむので，
発火率モデル→spikingモデルとしたほうが全体の流れとしてはよいか？
introも省略していいと思う．時間がない．

neuronの形態の話は別で節を作ろう．

ニューロンとシナプスの生理
発火率モデル
local-learning-rule
energy based
solve-credit-assignment-problem

やはりこれを前提としたモデルの構築が必要なので，発火率ですむ部分は前に持っていく．

第n章 Spiking neural network
ぐらいにしておいてもいいかも．
第1節. ニューロンのモデル
モデルを列挙…

neuron-model
synapse-model
neuronal-computation

リザバーコンピューティング (発火率・spiking)
ベイズ推論
運動学習
強化学習
ネットワーク・形態学


## Introduction
### intro
### computational-neuroscience
### notation
### usage-julia-lang
### linear-algebra
### differential-equation
### linear-regression
### probability-information-theory
### stochastic-process-differential-equation

## neuron-model
### neuron-physiol
### hodgkin-huxley
### fhn
### lif
### izhikevich
### isi
### neurite-growth-model

## synapse-model
### synapse-physiol
### current-conductance-synapse
### expo-synapse
### kinetic-synapse
### synaptic-weighted
### dynamical-synapses

## neuronal-computation
  sections:
  - file: neuronal-computation/neuronal-arithmetic
- file: local-learning-rule/intro
  sections:
  - file: local-learning-rule/pca-hebbian-learning
  - file: local-learning-rule/mds-anti-hebbian-learning
  - file: local-learning-rule/slow-feature-analysis
  - file: local-learning-rule/stdp-learning
  - file: local-learning-rule/logistic-regression-perceptron
  - file: local-learning-rule/self-organizing-map
  - file: local-learning-rule/heavy-tail

## energy-based-model
  sections:
  - file: energy-based-model/energy-based-model
  - file: energy-based-model/hopfield-model
  - file: energy-based-model/boltzmann-machine
  - file: energy-based-model/sparse-coding
  - file: energy-based-model/predictive-coding

## solve-credit-assignment-problem
  sections:
  - file: solve-credit-assignment-problem/backpropagation
  - file: solve-credit-assignment-problem/linear-network-learning-dynamics
  - file: solve-credit-assignment-problem/bptt
  - file: solve-credit-assignment-problem/surrogate-gradient-snn
  - file: solve-credit-assignment-problem/reservoir-computing

## motor-learning
  sections:
  - file: motor-learning/minimum-jerk
  - file: motor-learning/minimum-variance
  - file: motor-learning/optimal-feedback-control
  - file: motor-learning/infinite-horizon-ofc
  - file: motor-learning/local-learning-ofc
  - file: motor-learning/rat-trajectory
## reinforcement-learning
  sections:
  - file: reinforcement-learning/td-learning
## bayesian-brain
  sections:
  - file: bayesian-brain/neural-uncertainty-representation
  - file: bayesian-brain/bayesian-linear-regression
  - file: bayesian-brain/mcmc
  - file: bayesian-brain/neural-sampling
  - file: bayesian-brain/probabilistic-population-coding
  - file: bayesian-brain/quantile-expectile-regression
- file: appendix/intro
  sections:
  - file: appendix/grid-cells-decoding
  - file: appendix/graph-theory-network-model
  - file: appendix/useful-links
  - file: appendix/usage-jupyter-book
