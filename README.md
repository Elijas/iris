#Summary
Welcome to my first Neural Network experimental project! Here, I compared performances of Neural Networks with different Network Architectures trained with different regularization parameters lambda.  
  
The algorithms and some support functions were taken from <a href="https://www.coursera.org/course/ml">Stanford's Machine Learning MOOC</a> videos and homework tasks, but I rewritten parts of it to work with Neural Networks containing any number of layers (and any number of neurons inside the layers).

Update: later on, I also got curious about actual NN output relation to individual feature values, but there is no straightforward visual way to do examine such things because of 4 dimensional feature space. I still got pretty satisfactory results by (A) plotting each possible pair of features against each other and (B) running PCA to extract 2 or 3 dimensional features.
 
#Info
- Dataset used: <a href=https://archive.ics.uci.edu/ml/datasets/Iris>Iris</a>  
- Software used: Octave 3.6.4  
- Contents/Usage:  
  - run* `nnTraining.nn` to train and compare Neural Networks of different sizes  
  - run `plotFeaturePairs.m` and `plotPCA.m` to plot features against trained NN output  
(* note: data is shuffled every time the program is run. To reproduce experiment results seen below (or run experiments with lambda), uncomment the line `load('irisDataPrepared')`)

#Output snapshots from nnTraining.m 
Update 2: (Added polynomial features (up to third degree), got better test set accuracy)
```
Training...  (lambda = 2.2, iteration limit = 200)
-----------------------------------
Layers  | Acc. (trn.) | Acc. (test)
 34 3   |      98.41% |     100.00%
 34 1   |      97.14% |     100.00%
-----------------------------------
```

Update 1: (got interested whether performance is limited by a lack of layers/neurons, so I ran several additional tests)
```
# Experiment No.4: Training a Large NN
Training...  (lambda = 0.5, iteration limit = 40000)
 4 8 8 8 8 3    |      98.73% |      97.04%
 4 150 3        |      98.73% |      97.04%
```

Experiments with different lambda parameters and NN architectures:
```
# Experiment No.1: Training a Single Layer Perceptron
Training...  (lambda = 0, iteration limit = 1000)
Accuracy: 75.5556% (predicting class 2, test set)
             
# Experiment No.2: Training Multi-layered Neural Networks
Training...  (lambda = 0, iteration limit = 3000)
-----------------------------------
Layers  | Acc. (trn.) | Acc. (test)
 4 3    |      91.43% |      90.37% 
 4 1 3  |      88.57% |      86.67% 
 4 2 3  |     100.00% |      93.33% 
 4 3 3  |     100.00% |      91.11% 
 4 4 3  |      99.37% |      95.56% 
 4 5 3  |     100.00% |      91.11% 
 4 6 3  |      98.73% |      97.04% 
 4 7 3  |     100.00% |      97.04% 
-----------------------------------
             
# Experiment No.3: Training Multi-layered NNs for separate classes
Training...  (lambda = 0, iteration limit = 3000)
-----------------------------------
Layers  | Acc. (trn.) | Acc. (test)
 4 1    |      97.14% |      93.33% 
 4 1 1  |      98.10% |      95.56% 
 4 2 1  |      98.10% |      93.33% 
 4 3 1  |     100.00% |      91.11% 
 4 4 1  |     100.00% |      91.11% 
 4 5 1  |     100.00% |      88.89% 
 4 6 1  |     100.00% |      88.89% 
 4 7 1  |     100.00% |      93.33% 
-----------------------------------
```
```
# Experiment No.1: Training a Single Layer Perceptron
Training...  (lambda = 0.1, iteration limit = 100)
Accuracy: 75.5556% (predicting class 2, test set)
             
# Experiment No.2: Training Multi-layered Neural Networks
Training...  (lambda = 0.1, iteration limit = 500)
-----------------------------------
Layers  | Acc. (trn.) | Acc. (test)
 4 3    |      91.43% |      90.37% 
 4 1 3  |      88.57% |      85.93% 
 4 2 3  |      98.73% |      97.04% 
 4 3 3  |      98.73% |      97.04% 
 4 4 3  |      98.73% |      97.04% 
 4 5 3  |      98.73% |      97.04% 
 4 6 3  |      98.73% |      97.04% 
 4 7 3  |      98.73% |      97.04% 
-----------------------------------
             
# Experiment No.3: Training Multi-layered NNs for separate classes
Training...  (lambda = 0.1, iteration limit = 500)
-----------------------------------
Layers  | Acc. (trn.) | Acc. (test)
 4 1    |      96.19% |      91.11% 
 4 1 1  |      97.14% |      91.11% 
 4 2 1  |      98.10% |      95.56% 
 4 3 1  |      98.10% |      95.56% 
 4 4 1  |      98.10% |      95.56% 
 4 5 1  |      98.10% |      95.56% 
 4 6 1  |      98.10% |      95.56% 
 4 7 1  |      98.10% |      95.56% 
-----------------------------------
```
```
# Experiment No.1: Training a Single Layer Perceptron
Training...  (lambda = 10, iteration limit = 100)
Accuracy: 68.8889% (predicting class 2, test set)
             
# Experiment No.2: Training Multi-layered Neural Networks
Training...  (lambda = 10, iteration limit = 500)
-----------------------------------
Layers  | Acc. (trn.) | Acc. (test)
 4 3    |      88.89% |      88.15% 
 4 1 3  |      66.67% |      66.67% 
 4 2 3  |      77.46% |      77.78% 
 4 3 3  |      77.46% |      77.78% 
 4 4 3  |      78.10% |      77.78% 
 4 5 3  |      78.10% |      77.78% 
 4 6 3  |      79.37% |      78.52% 
 4 7 3  |      79.37% |      78.52% 
-----------------------------------
             
# Experiment No.3: Training Multi-layered NNs for separate classes
Training...  (lambda = 10, iteration limit = 500)
-----------------------------------
Layers  | Acc. (trn.) | Acc. (test)
 4 1    |      88.57% |      84.44% 
 4 1 1  |      68.57% |      66.67% 
 4 2 1  |      34.29% |      31.11% 
 4 3 1  |      68.57% |      66.67% 
 4 4 1  |      68.57% |      71.11% 
 4 5 1  |      69.52% |      71.11% 
 4 6 1  |      69.52% |      71.11% 
 4 7 1  |      69.52% |      71.11% 
-----------------------------------
```
