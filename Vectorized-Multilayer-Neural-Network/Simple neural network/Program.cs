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

            for (int epoch = 0; epoch < 30001; epoch++)
            {
                //FORWARDPROPAGATION
                Matrix[] Z, A;
                ForwardPropagation(out Z, out A, W, ExampleCount, LayerCount, InputValue);

                Matrix Zlast = Z[LayerCount - 1].Slice(0, 1, Z[LayerCount - 1].x, Z[LayerCount - 1].y);
                Matrix output = A[A.Length - 1].Slice(0, 1, A[A.Length - 1].x, A[A.Length - 1].y);

                //BACKPROPAGATION
                Matrix[] error, delta;
                BackPropagation(out delta, out error, output, OutputValue, Zlast, W, Z, A, LayerCount);
                Matrix LastError = error[LayerCount - 1];

                //Gradient Descend
                GradientDescend(ref W, A, delta, LearningRate);

                Console.WriteLine("Loss: " + LastError.abs.average * ExampleCount);
                if (epoch % 1000 == 0)
                {
                    Console.WriteLine("-------" + epoch + "----------------");
                    Console.WriteLine(InputValue);
                    Console.WriteLine(output);
                }

            }
            Console.Read();
        }
        static void ForwardPropagation(out Matrix[] Z, out Matrix[] A, Matrix[] W,
                                        int ExampleCount, int LayerCount, Matrix InputValue)
        {
            Z = new Matrix[LayerCount];
            A = new Matrix[LayerCount];

            Z[0] = InputValue.AddColumn(Matrix.Ones(ExampleCount, 1));
            A[0] = Z[0];

            for (int i = 1; i < LayerCount; i++)
            {
                Z[i] = (A[i - 1] * W[i - 1]).AddColumn(Matrix.Ones(ExampleCount, 1));
                A[i] = sigmoid(Z[i]);//Relu(Z[i]); <- Uncomment if Relu
            }
            //A[A.Length - 1] = Z[Z.Length - 1]; <- Uncomment if relu OR iregularized Values
        }
        static void BackPropagation(out Matrix[] delta, out Matrix[] error,Matrix output, Matrix OutputValue,
                                    Matrix Zlast, Matrix[]W, Matrix[]Z, Matrix[] A, int LayerCount)
        {
            error = new Matrix[LayerCount];
            error[LayerCount - 1] = output - OutputValue;
            delta = new Matrix[LayerCount];
            delta[LayerCount - 1] = error[LayerCount - 1] * sigmoid(Zlast, true);

            for (int i = LayerCount - 2; i >= 0; i--)
            {
                error[i] = delta[i + 1] * W[i].T;
                delta[i] = error[i] * sigmoid(Z[i], true);
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
        static Matrix sigmoid(Matrix m, bool derivated = false)
        {
            double[,] output = m;
            Matrix.MatrixLoop((i, j) => {
                if (derivated)
                {
                    double aux = 1 / (1 + Math.Exp(-output[i, j]));
                    output[i, j] = aux * (1 - aux);
                }
                else
                {
                    output[i, j] = 1 / (1 + Math.Exp(-output[i, j]));
                }

            }, m.x, m.y);
            return output;
        }
        static Matrix Relu(Matrix m, bool derivated = false)
        {
            double[,] output = m;
            Matrix.MatrixLoop((i, j) => {
                if (derivated)
                {                    
                    output[i, j] = output[i, j] > 0 ? 1 : 0;
                }
                else
                {
                    output[i, j] = output[i, j] > 0 ? output[i, j] : 0;
                }

            }, m.x, m.y);
            return output;
        }

    }
}
