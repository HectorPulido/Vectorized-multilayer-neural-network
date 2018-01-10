# Vectorized Multilayer Neural Network

This is a simple MultiLayer perceptron made with [Simple Linear Algebra for C#](https://github.com/HectorPulido/Simple_Linear_Algebra) , is a neural network based on [This Algorithm](https://github.com/HectorPulido/Simple-vectorized-mono-layer-perceptron) but generalized. This neural network can calcule logic doors like Xor Xnor And Or via Stochastic gradient descent backpropagation with Sigmoid as Activation function, but can be used to more complex problems.

There is a lot to improve, like csv read, gpu implementation, regularization, but is functional.
 
## How use it
Just go to the project and open Program.cs and run it, you can change the dataset changing X and Y variables, and choose the number of neurons and layers you want

## How use Relu
0. Change all sigmoid function, for relu function
1. Last A must have no Nonlinear function Matrix Last A must be Equal To Last Z;
2. because of that Last Delta has not derivated Matrix "Last Delta = Last error Error * 1";
3. The learning rate must be smaller, like 0.001 
4. Optionaly you can use a Softmax layer to make a clasifier

## Where can i learn more
On my Youtube channel (spanish) are a lot of information about Machine learning and Neural networks
https://www.youtube.com/channel/UCS_iMeH0P0nsIDPvBaJckOw
You can also look at the previous Example of This 
https://github.com/HectorPulido/Simple-vectorized-mono-layer-perceptron
Or Look at a Non Vectorized Example
https://github.com/HectorPulido/Multi-layer-perceptron

## Patreon
Please consider Support on Patreon
https://www.patreon.com/HectorPulido

