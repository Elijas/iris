#Summary:
My First Neural Network implementation. All the algorithms and other code were taken from Coursera Stanford Machine Learning MOOC videos and homework assignments. I wanted to compare performances of different NN architectures on the same task so I changed the algorithms of Forward and Back propagations to work with Neural Network of any number of layers and elements inside them. 

#Info
Dataset used: iris
Software used: Octave
Instructions: run the "main.m" file

#Experiment snapshot
Experiment No.1: Using three Single Layer Perceptrons for classification:
Accuracies of the test set (lambda = 0):
Class 1: 100%
Class 2: 77%
Class 3: 97%

Experiment No.2: Using a Multi-layered Neural Network:
Accuracies (lambda = 0.001):
90.7937 (tr.set)  90.3704 (test set)  | Layers:   4   3
78.0952 (tr.set)  77.0370 (test set)  | Layers:   4   1   3
98.7302 (tr.set)  100.0000 (test set) | Layers:   4   2   3
99.3651 (tr.set)  100.0000 (test set) | Layers:   4   3   3
100.0000 (tr.set) 100.0000 (test set) | Layers:   4   4   3
100.0000 (tr.set) 100.0000 (test set) | Layers:   4   5   3
100.0000 (tr.set) 100.0000 (test set) | Layers:   4   6   3
100.0000 (tr.set) 100.0000 (test set) | Layers:   4   7   3

Experiment No.3: Using separate Multi-layered Neural Networks for separate classes:
Accuracies (lambda = 0.001):
97.1429 (tr.set)  100.0000 (test set) | Layers:   4   1
98.0952 (tr.set)  100.0000 (test set) | Layers:   4   1   1
98.0952 (tr.set)  100.0000 (test set) | Layers:   4   2   1
100.0000 (tr.set) 100.0000 (test set) | Layers:   4   3   1
100.0000 (tr.set) 100.0000 (test set) | Layers:   4   4   1
100.0000 (tr.set) 97.7778 (test set)  | Layers:   4   5   1
100.0000 (tr.set) 100.0000 (test set) | Layers:   4   6   1
100.0000 (tr.set) 100.0000 (test set) | Layers:   4   7   1

