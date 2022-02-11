# ottergrad

![ottergrad Logo](https://github.com/raulniconico/ottergrad/blob/0f4b8dad0bf5052cfb002ddb3a2ce56731263634/logo/ottergrad.png)

Ottergrad is an automatic differentiation tool support plenty of NumPy functions who borns from [Nuwa framework](https://github.com/raulniconico/Nuwa0.0.1).
This project separates the auto-derivative function from Nuwa into a package, whose algorithm is more efficient, simpler and more stable than Nuwa 0.0.2.


First of all, the API is very simple, all functions and classes are of NumPy type. Function calls and arguments are the same as in NumPy. So for projects using NumPy, ottergrad can be migrated very easily and don't need any extra API learning.

Second, ottergrad's computation graph can be easily used in Deep learning training. There are fleurish activation functions and loss functions which are widely used.

The following are the main components of ottergrad and a mini user guide, before using it, execute the following line:

        pip install -i https://test.pypi.org/simple/ Ottergrad

<!-- toc -->

- [Tensor](#Tensor)
- [Numpy support](#Numpy-support)
- [Func and Graph](#Func-and-Graph)
- [Forward and back propagation](#Forward-and-back-propagation)
- [Neural Network support](#Neural-Network-support)
- [Examples](#Examples)

<!-- tocstop -->


### Tensor
Every calculation in ottergrad is Tensor operation
One can simply create a Tensor instance by firstly import Ottergrad.autograd.Tensor

    from Ottergrad.autograd import Tensor
    W = Tensor(np.ones((1,5)))
    x = Tensor(np.ones((5,3)))

As shown in the code above, we can create two instances of Tensor, W and x, just like input and weight in the neural network linear layer.
The Tensor does not need to be assigned a value at initialization but can be created first for the calculation graph. One can use setdata() to assign before forward propagation.
    
    y = Tensor()
    y.setdata(np.zero((3, 3)))

### Numpy support
ottergrad provides a huge number of NumPy type operaters. Use following line to get all operators:
    
    Tensor().OVERLOADABLE_OPERATORS

For using these operators, we import otternumpy, then use them just like NumPy operators

    import Ottergrad.otternumpy as otnp
    otnp.dot(W, x)

Instead of computing the operator value directly, otternumpy will retrieve a Tensor instance and use it as a node in the calculation graph. Notice that, in the case of neural network, we consider a linear layer like following:

    Sigmoid(otnp.dot(W, x))


### Func and Graph

Once we have operators, we start to string them together into a whole calculation flow and compute and derive them in turn by the method of graph. So we use the Func class, which takes Tensor objects as inputs and builds graphs from them. Calculation details are explained in https://github.com/raulniconico/Nuwa0.0.1/blob/master/Nuwa_framework.pdf
    
    func = Sigmoid(otnp.dot(W, x))
    func = Func(func)


### Forward and back propagation
Func class provides forward and backword, first forward propagation of the function, the method will return the final output

    func.forward()
    array([[0.99330715, 0.99330715, 0.99330715]]
    
Then use the backward() method of func to auto-derive the inputs and parameters, the derivative of each variable can be obtained from the getgrad() of the variable

    func.backward()
    W.getgrad()
    array([[0.02035004, 0.02035004, 0.02035004, 0.02035004, 0.02035004]])
    
### Neural Network support
ottergrad provides the most commonly used activation functions, loss functions and optimization algorithms used in neural networks.

    from Ottergrad.activation import Sigmoid, ReLU
    from Ottergrad.loss import MSE

and so on. They can be use as function.
### Examples
You can visit https://github.com/raulniconico/ottergrad/blob/0f4b8dad0bf5052cfb002ddb3a2ce56731263634/test/demo.ipynb to try the whole demo or use your own function,
Another example is using otnp.where function as follow:

    a = Tensor()
    a.setdata(np.array([[1, 2, 3], [4, 5, 6]]))
    func = Func(otnp.where(a > 3, 3 * a, 0))
    func.forward()
    func.backward()
    a.getgrad()

Finally, we give a vivid 3 layers fully connected network to see if the gradient can be correctly given

    x = Tensor(np.random.rand(5000, 30))
    W1 = Tensor(np.random.rand(30,100))
    W2 = Tensor(np.random.rand(100, 50))
    W3 = Tensor(np.random.rand(50, 30))
    
    l1 = Sigmoid(otnp.dot(x, W1))
    l2 = Sigmoid(otnp.dot(l1, W2))
    l3 = otnp.dot(l2, W3)
    func = Func(l3)
    
    func.forward()
    func.backward()
    
    print("W1's gradient is " + str(W1.getgrad()))

    W1's gradient is [[1.55841779 3.09941541 2.56407321 ... 4.21642738 2.98262384 2.19862303]
     [1.49618532 3.07101007 2.50246591 ... 3.72816309 3.44169822 2.14018203]
     [1.57303764 2.91315111 2.5995923  ... 4.00325851 3.29807133 2.03043232]
     ...
     [1.4124521  3.21118696 2.62250154 ... 3.68704166 3.28038238 1.98191629]
     [1.5642053  2.93563087 2.7684254  ... 3.71760803 3.13955798 2.06892669]
     [1.42228891 3.08226813 2.89191136 ... 3.50635962 3.0772686  1.92641901]]