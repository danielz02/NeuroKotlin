# Neural Network With Kotlin
Author: Chenhui Zhang && Yihong Jian

In this project, [my partner](https://github.com/yihjian/) and I implemented the key components of a simple neural network and assembled a model with fully connected layers.

We used Kotlin-Numpy, a language binding of Python's Numpy library, to deal with the majority of tensor operations.

LICENSE: [anti-996](https://github.com/996icu/996.ICU).

# Works Done

We've modularized all of our code so that all the modules are extendable for future works. Below is a list of what we've done.

* Activation Functions: We've implemented ReLU, sigmoid, and hyperbolic tangent and their gradients.
* Layers: We've done with the basic linear layer including its forward path and backward path. I'll implement the convolution layer this weekend.
* Models: We've assembled a model with fully connected layers but there's still some work to be done with backprop.
* Optimizers: SGD and Adam should be good for now.
* Schedulers: Constant scheduler and exponential scheduler for learning rate decay.
[This is a link to our repo.](https://github.com/danielz02/NeuroKotlin)

## Works Done

We've modularized all of our code so that all the modules are extendable for future works. Below is a list of what we've done.

* Activation Functions: We've implemented ReLU, sigmoid, and hyperbolic tangent and their gradients.
* Layers: We've done with the basic linear layer including its forward path and backward path. I'll implement the convolution layer this weekend.
* Models: We've assembled a model with fully connected layers but there's still some work to be done with backprop.
* Optimizers: SGD and Adam should be good for now.
* Schedulers: Constant scheduler and exponential scheduler for learning rate decay.
* UtilsL L1 and L2 distance. We'll also implement convolution operation and data preprocessing here.

* Utils: L1 and L2 distance. We'll also implement convolution operation and data preprocessing here.

In general, working with Kotlin is a great experience. It saved me a lot of code for null checking and other boilerplates. Blending functional style code into an object oriented design helped me make the code cleaner and I haven't written any loop in this project.

## Problems

Below are some problems we've encountered.

* Python dependency: This is a huge pain initially because I have two system-wide python environments in `/usr/bin` and one conda environment. This library depends on Python 3 but it looks for `pip` and `python`, which will call my system's Python 2 and then fail. I ended up manually overriding the PATH environmental variables in IDEA's run configuration to my conda environment.
* Changing the mindset: In Kotlin we can't really use Python mindset of "let's execute and see what happens."
* Calculating gradients: It's a huge pain since we're doing from scratch and AutoGrad is far beyond the current scope of this project.
* Don't hava direct access to Numpy's numerical datatypes like np.inf and memory management is done by Python.
* No documentation for NumKt: Looking up Numpy's documentation while writing Kotlin code feels weird and Numpy's Kotlin wrapper doesn't provide access to some functions like `np.apply_along_axis`. There's an `apply` method on `KtNdArray` objects but the function passed to this method is called on each slice of `KtNdArray` without providing access to `it`. I resolved this issue with Kotlin's extension functions.
* Variable length arguments: Kotlin doesn't have `**kwargs`. It has `vararg` keyword for variable length arguments but such function can't be overriden by its children. This could be a problem for high dimension tensors when we need to pass its shape as an argument since Kotlin doesn't have n-tuple.

## Future Works

* Demo in Jupyter Notebook.
* Convolution operations, padding, and pooling layers.
* Batch normalizations, regularization (weight decay), early stopping, and dropout.
* Data preprocessing pipelines and dataloader with concurrency.
* Glorot weight initializers.
* Simple AutoGrad.

## Acknowledgement

* [Numpy-ml](https://github.com/ddbourgin/numpy-ml/): Inspiration of class design 
* [Stanford's CS 231n](https://cs231n.github.io/): Everything