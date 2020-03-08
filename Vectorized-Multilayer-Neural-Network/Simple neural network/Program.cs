using System;
using LinearAlgebra;

/*
 THIS PROGRAM USES SIMPLE LINEAR ALGEBRA LIBRARY FROM GITHUB
 https://github.com/HectorPulido/Simple_Linear_Algebra     
*/

namespace VectorizedMultiLayerPerceptron
{
    class Program
    {
        static void Main(string[] args)
        {

            var mlp = new VectorizedMatrixMLP();

            mlp.InputValue = new double[,] {
                {1, 0},
                {0, 0},
                {0, 1},
                {1, 1}};
            mlp.TargetValue = new double[,]{ 
                {1},
                {0},
                {1},
                {0}};

            //mlp.activFct = new ActivationFunction.SigmoidFunction();
            //mlp.nbIterations = 10000; // Sigmoid: works

            //mlp.activFct = new ActivationFunction.HyperbolicTangentFunction();
            //mlp.nbIterations = 5000; // Hyperbolic tangent: works

            //mlp.activFct = new ActivationFunction.ELUFunction();
            //mlp.nbIterations = 1000; // ELU: works fine
                        
            //mlp.activFct = new ActivationFunction.GaussianFunction();
            //mlp.nbIterations = 1000; // Gaussian: works fine
                        
            //mlp.activFct = new ActivationFunction.SinusFunction();
            //mlp.nbIterations = 500; // Sinus: works fine
                        
            //mlp.activFct = new ActivationFunction.ArcTangentFunction();
            //mlp.nbIterations = 1000; // ArcTangent: works fine
                        
            //mlp.activFct = new ActivationFunction.ReluFunction();
            //mlp.nbIterations = 100000; // ReLU: Does not work yet, but this next one yes:
                        
            //mlp.activFct = new ActivationFunction.ReLUSigmoidFunction();
            //mlp.nbIterations = 5000; // ReLUSigmoid: works fine

            mlp.lambdaFct = (x) => mlp.activFct.Activation(x, gain: 1, center: 0);
            mlp.lambdaFctD = (x) => mlp.activFct.Derivative(x, gain: 1, center: 0);
            mlp.activFctIsNonLinear = mlp.activFct.IsNonLinear();

            //int[] NeuronCount = new int[] { 2, 3, 3, 1 };
            int[] NeuronCount = new int[] { 2, 2, 1 };
            mlp.LearningRate = 0.1;
            mlp.InitStruct(NeuronCount, addBiasColumn: true);

            mlp.Randomize();

            mlp.Train();
                        
            Console.WriteLine("Press a key to quit.");
            Console.Read();
        }
    }
}
