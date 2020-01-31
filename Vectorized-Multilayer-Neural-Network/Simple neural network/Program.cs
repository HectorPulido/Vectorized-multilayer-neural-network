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
            Matrix InputValue = new double[,] { { 0, 0 },
                                                { 0, 1 },
                                                { 1, 0 },
                                                { 1, 1 } };
            Matrix OutputValue = new double[,]{ { 0 },
                                                { 1 },
                                                { 1 },
                                                { 0 } };

            int nbIter;

            nbIter = 5000; // Sigmoid: works
            var activFct = new ActivationFunction.SigmoidFunction();

            //nbIter = 5000; // Hyperbolic tangent: works
            //var activFct = new ActivationFunction.HyperbolicTangentFunction();

            //nbIter = 10000; // ELU: Does not work yet
            //var activFct = new ActivationFunction.ELUFunction();

            //nbIter = 1000; // Gaussian: works fine
            //var activFct = new ActivationFunction.GaussianFunction();

            //nbIter = 100000; // Sinus: works fine normally, but not there !?
            //var activFct = new ActivationFunction.SinusFunction();

            //nbIter = 1000; // ArcTangent: works fine
            //var activFct = new ActivationFunction.ArcTangentFunction();

            //nbIter = 100000; // ReLU: Does not work yet, but the next one yes
            //var activFct = new ActivationFunction.ReluFunction();

            //nbIter = 5000; // ReLUSigmoid: works fine
            //var activFct = new ActivationFunction.ReluSigmoidFunction();

            Func<double, double> lambdaFct = (x) => activFct.Activation(x, gain: 1, center: 0);
            Func<double, double> lambdaFctD = (x) => activFct.Derivative(x, gain: 1, center: 0);
            bool activFctIsNonLinear = activFct.IsNonLinear();

            int[] NeuronCount = new int[] { 2, 3, 3, 1 };
            int LayerCount = NeuronCount.Length;
            int ExampleCount = OutputValue.x;

            double LearningRate = 0.8;

            Random r = new Random(1);

            Matrix[] W = new Matrix[LayerCount - 1];
            for (int i = 0; i < W.Length; i++)
            {
                W[i] = Matrix.Random(NeuronCount[i] + 1, NeuronCount[i + 1], r) * 2 - 1;
            }

            for (int epoch = 0; epoch < nbIter; epoch++)
            {
                //FORWARDPROPAGATION
                Matrix[] Z, A;
                ForwardPropagation(out Z, out A, W, ExampleCount, LayerCount, InputValue, 
                    lambdaFct, activFctIsNonLinear);

                Matrix Zlast = Z[LayerCount - 1].Slice(0, 1, Z[LayerCount - 1].x, Z[LayerCount - 1].y);
                Matrix output = A[A.Length - 1].Slice(0, 1, A[A.Length - 1].x, A[A.Length - 1].y);

                //BACKPROPAGATION
                Matrix[] error, delta;
                BackPropagation(out delta, out error, output, OutputValue, Zlast, W, Z, A, LayerCount, lambdaFctD);
                Matrix LastError = error[LayerCount - 1];

                //Gradient Descend
                GradientDescend(ref W, A, delta, LearningRate);
                
                if ((epoch < 10) ||
                    (((epoch + 1) % 10000) == 0) ||
                    ((((epoch + 1) % 1000) == 0) && epoch < 10000))
                {
                    Console.WriteLine("Loss: " + LastError.abs.average * ExampleCount);
                    Console.WriteLine("-------" + (epoch+1) + "/" + nbIter + 
                        "----------------");
                    Console.WriteLine(InputValue);
                    Console.WriteLine(output);
                }

            }
            Console.WriteLine("Press a key to quit.");
            Console.Read();
        }
        static void ForwardPropagation(out Matrix[] Z, out Matrix[] A, Matrix[] W,
            int ExampleCount, int LayerCount, Matrix InputValue, 
            Func<double, double> lambdaFct, bool activFctIsNonLinear)
        {
            Z = new Matrix[LayerCount];
            A = new Matrix[LayerCount];

            Z[0] = InputValue.AddColumn(Matrix.Ones(ExampleCount, 1));
            A[0] = Z[0];

            for (int i = 1; i < LayerCount; i++)
            {
                Z[i] = (A[i - 1] * W[i - 1]).AddColumn(Matrix.Ones(ExampleCount, 1));
                A[i] = Matrix.Map(Z[i], lambdaFct);
                //A[i] = sigmoid(Z[i]);//Relu(Z[i]); <- Uncomment if Relu
            }
            if (activFctIsNonLinear) A[A.Length - 1] = Z[Z.Length - 1]; //<- Uncomment if relu OR iregularized Values
        }
        static void BackPropagation(out Matrix[] delta, out Matrix[] error,
            Matrix output, Matrix OutputValue,
            Matrix Zlast, Matrix[]W, Matrix[]Z, Matrix[] A, int LayerCount,
            Func<double, double> lambdaFctD)
        {
            error = new Matrix[LayerCount];
            error[LayerCount - 1] = output - OutputValue;
            delta = new Matrix[LayerCount];
            //delta[LayerCount - 1] = error[LayerCount - 1] * sigmoid(Zlast, true);
            delta[LayerCount - 1] = error[LayerCount - 1] * Matrix.Map(Zlast, lambdaFctD);

            for (int i = LayerCount - 2; i >= 0; i--)
            {
                error[i] = delta[i + 1] * W[i].T;
                //delta[i] = error[i] * sigmoid(Z[i], true);
                delta[i] = error[i] * Matrix.Map(Z[i], lambdaFctD);
                delta[i] = delta[i].Slice(0, 1, delta[i].x, delta[i].y);
            }
        }
        static void GradientDescend(ref Matrix[] W, Matrix[] A,
                            Matrix[] delta, double LearningRate)
        {
            for (int i = 0; i < W.Length; i++)
            {
                W[i] -= (A[i].T * delta[i + 1]) * LearningRate;
            }
        }
        //static Matrix sigmoid(Matrix m, bool derivated = false)
        //{
        //    double[,] output = m;
        //    Matrix.MatrixLoop((i, j) => {
        //        if (derivated)
        //        {
        //            double aux = 1 / (1 + Math.Exp(-output[i, j]));
        //            output[i, j] = aux * (1 - aux);
        //        }
        //        else
        //        {
        //            output[i, j] = 1 / (1 + Math.Exp(-output[i, j]));
        //        }

        //    }, m.x, m.y);
        //    return output;
        //}
        //static Matrix Relu(Matrix m, bool derivated = false)
        //{
        //    double[,] output = m;
        //    Matrix.MatrixLoop((i, j) => {
        //        if (derivated)
        //        {                    
        //            output[i, j] = output[i, j] > 0 ? 1 : 0;
        //        }
        //        else
        //        {
        //            output[i, j] = output[i, j] > 0 ? output[i, j] : 0;
        //        }

        //    }, m.x, m.y);
        //    return output;
        //}

    }
}
