MATLAB package for Deep Canonically Correlated Autoencoders (DCCAE)

(C) 2015 by Weiran Wang, Raman Arora, Karen Livescu and Jeff Bilmes

This Matlab code implements the Deep Canonically Correlated Autoencoders 
  (DCCAE) algorithm described in the paper:

  Weiran Wang, Raman Arora, Karen Livescu, and Jeff Bilmes. 
  On Deep Multi-View Representation Learning. 
  The 32nd International Conference on Machine Learning (ICML 2015).

Quick start:
- demo.m: demonstrates the usage of linear/kernel/deep CCA/DCCAE on noisy
  MNIST images (random rotation and background) as two views.

List of functions:
- DCCA_grad.m: computes the gradient of total correlation with respect to 
  the neural network weights for each view.
- DCCA_train.m: DCCA training with stochastic minibatch optimization.
- DCCAE_train.m: DCCAE training with stochastic minibatch optimization.
- linCCA.m: linear CCA.
- randKCCA.m: randomized kernel CCA, proposed in 
  David Lopez-Paz, Suvrit Sra, Alex Smola, Zoubin Ghahramani, and 
  Bernhard Schoelkopf.
  Randomized Nonlinear Component Analysis.
  The 31th International Conference on Machine Learning (ICML) 2014.
- svmclassify.m: one-versus-one linear SVM by calling libsvm. The libsvm
  package is available from http://www.csie.ntu.edu.tw/~cjlin/libsvm/.
- createMNIST.m: generates the MNIST images used in demo.m (the random 
  seed used to generate our data is saved in demoseed.mat, so that you can 
  achieve exactly the same result).

The following are used internally by other functions (in ./deepnet)
- deepnetfwd.m: feed-forwards the inputs through deep neural networks to 
  obtain outputs.
- deepnetinit.m: initializes the neural network weight parameters.
- deepnetgrad.m: computes gradient of reconstruction error with respect to 
  neural network weights.

External packages/data:
- mnist_all.mat: the MATLAB format of all MNIST images, can be downloaded 
  from Sam Roweis's webpage http://www.cs.nyu.edu/~roweis/data.html.
- RBMPRETRAIN_K=10.mat: pretrained deep autocencoders using restricted 
  Boltzmann machines (RBMs), we use the following code for pretraining:
  http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html
  Reference: 
    G. E. Hinton and R. R. Salakhutdinov. 
    Reducing the Dimensionality of Data with Neural Networks.
    Science, 2006.
