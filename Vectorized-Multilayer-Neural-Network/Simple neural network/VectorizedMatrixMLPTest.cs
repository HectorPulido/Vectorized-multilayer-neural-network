using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using LinearAlgebra;

namespace VectorizedMultiLayerPerceptron
{
    [TestClass()]
    public class VecMatrixMLPTest
    {
        private VectorizedMatrixMLP m_mlp;
        private int[] NeuronCount;

        [TestInitialize()]
        public void Init()
        {
            m_mlp = new VectorizedMatrixMLP();
            NeuronCount = new int[] {2, 2, 1};
            m_mlp.LearningRate = 0.1;
            m_mlp.InputValue = new double[,] { 
                {1, 0},
                {0, 0},
                {0, 1},
                {1, 1}};

            m_mlp.TargetValue = new double[,] { 
                {1},
                {0},
                {1},
                {0}};
        }

        [TestMethod()]
        public void VectorizedMatrixMLPXORSigmoid()
        {
            m_mlp.nbIterations = 30000; // Sigmoid: works
            m_mlp.activFct = new ActivationFunction.SigmoidFunction();
            m_mlp.lambdaFct = x => m_mlp.activFct.Activation(x, gain: 1, center: 0);
            m_mlp.lambdaFctD = x => m_mlp.activFct.Derivative(x, gain: 1, center: 0);
            m_mlp.activFctIsNonLinear = m_mlp.activFct.IsNonLinear();

            m_mlp.InitStruct(NeuronCount, addBiasColumn: true);

            m_mlp.W[0] = new double[,] { 
                {-0.5, -0.78, -0.07}, 
                {0.54, 0.32, -0.13}, 
                {-0.29, 0.89, -0.8}};
            m_mlp.W[1] = new double[,] { 
                {0.28}, 
                {-0.94}, 
                {-0.5}, 
                {-0.36}};

            m_mlp.Train();

            var expectedOutput = new double[,] { 
                {0.97},
                {0.02},
                {0.97},
                {0.02}};

            var output = m_mlp.output.ToString();
            Matrix expectedMatrix = expectedOutput; // double(,) -> Matrix
            var expectedOutputString = expectedMatrix.ToString();
            Assert.AreEqual(output, expectedOutputString);

            var expectedLoss = 0.1;
            float[] targetArray = m_mlp.TargetValue.ToArray();
            var loss = m_mlp.ComputeAverageError(targetArray);
            var lossRounded = Math.Round(loss, 2);
            Assert.AreEqual(expectedLoss, lossRounded);
        }

        [TestMethod()]
        public void VectorizedMatrixMLPXORSigmoidWithoutBias()
        {
            m_mlp.nbIterations = 100000; // Sigmoid: works
            m_mlp.activFct = new ActivationFunction.SigmoidFunction();
            m_mlp.lambdaFct = x => m_mlp.activFct.Activation(x, gain: 1, center: 0);
            m_mlp.lambdaFctD = x => m_mlp.activFct.Derivative(x, gain: 1, center: 0);
            m_mlp.activFctIsNonLinear = m_mlp.activFct.IsNonLinear();

            m_mlp.InitStruct(NeuronCount, addBiasColumn: false);

            m_mlp.W[0] = new double[,] { 
                {0.38, 0.55},
                {0.24, 0.58}};
            m_mlp.W[1] = new double[,] { 
                {0.2},
                {0.16}};

            m_mlp.Train();

            var expectedOutput = new double[,] { 
                {0.93},
                {0.03},
                {0.93},
                {0.09}};

            var output = m_mlp.output.ToString();
            Matrix expectedMatrix = expectedOutput; // double(,) -> Matrix
            var expectedOutputString = expectedMatrix.ToString();
            Assert.AreEqual(output, expectedOutputString);

            var expectedLoss = 0.26;
            float[] targetArray = m_mlp.TargetValue.ToArray();
            var loss = m_mlp.ComputeAverageError(targetArray);
            var lossRounded = Math.Round(loss, 2);
            Assert.AreEqual(expectedLoss, lossRounded);
        }

        [TestMethod()]
        public void VectorizedMatrixMLPXORHTangent()
        {
            m_mlp.nbIterations = 5000; // Hyperbolic tangent: works
            m_mlp.activFct = new ActivationFunction.HyperbolicTangentFunction();
            m_mlp.lambdaFct = x => m_mlp.activFct.Activation(x, gain: 1, center: 0);
            m_mlp.lambdaFctD = x => m_mlp.activFct.Derivative(x, gain: 1, center: 0);
            m_mlp.activFctIsNonLinear = m_mlp.activFct.IsNonLinear();

            m_mlp.InitStruct(NeuronCount, addBiasColumn: true);

            m_mlp.W[0] = new double[,] { 
                {-0.5, -0.78, -0.07},
                {0.54, 0.32, -0.13},
                {-0.29, 0.89, -0.8}};
            m_mlp.W[1] = new double[,] {
                {0.28},
                {-0.94},
                {-0.5},
                {-0.36}};

            m_mlp.Train();

            var expectedOutput = new double[,] {
                {0.99},
                {0.0},
                {0.99},
                {0.0}};

            var output = m_mlp.output.ToString();
            Matrix expectedMatrix = expectedOutput; // double(,) -> Matrix
            var expectedOutputString = expectedMatrix.ToString();
            Assert.AreEqual(output, expectedOutputString);

            var expectedLoss = 0.02;
            float[] targetArray = m_mlp.TargetValue.ToArray();
            var loss = m_mlp.ComputeAverageError(targetArray);
            var lossRounded = Math.Round(loss, 2);
            Assert.AreEqual(expectedLoss, lossRounded);
        }

        [TestMethod()]
        public void VectorizedMatrixMLPXORELU()
        {
            m_mlp.nbIterations = 400; // ELU: works
            m_mlp.activFct = new ActivationFunction.ELUFunction();
            m_mlp.lambdaFct = x => m_mlp.activFct.Activation(x, gain: 0.1, center: 0.4);
            m_mlp.lambdaFctD = x => m_mlp.activFct.Derivative(x, gain: 0.1, center: 0.4);
            m_mlp.activFctIsNonLinear = m_mlp.activFct.IsNonLinear();
            m_mlp.LearningRate = 0.07;

            m_mlp.InitStruct(NeuronCount, addBiasColumn: true);

            m_mlp.W[0] = new double[,] {
                {0.37, 0.08, 0.74},
                {0.59, 0.54, 0.32},
                {0.2, 0.78, 0.01}};
            m_mlp.W[1] = new double[,] {
                {0.5},
                {0.27},
                {0.4},
                {0.11}};

            m_mlp.Train();

            var expectedOutput = new double[,] {
                {0.99},
                {0.02},
                {0.99},
                {0.01}};

            var output = m_mlp.output.ToString();
            Matrix expectedMatrix = expectedOutput; // double(,) -> Matrix
            var expectedOutputString = expectedMatrix.ToString();
            Assert.AreEqual(output, expectedOutputString);

            var expectedLoss = 0.04;
            float[] targetArray = m_mlp.TargetValue.ToArray();
            var loss = m_mlp.ComputeAverageError(targetArray);
            var lossRounded = Math.Round(loss, 2);
            Assert.AreEqual(expectedLoss, lossRounded);
        }

        [TestMethod()]
        public void VectorizedMatrixMLPXORGaussian()
        {
            m_mlp.nbIterations = 400; // Gaussian: works
            m_mlp.activFct = new ActivationFunction.GaussianFunction();
            m_mlp.lambdaFct = x => m_mlp.activFct.Activation(x, gain: 1, center: 0);
            m_mlp.lambdaFctD = x => m_mlp.activFct.Derivative(x, gain: 1, center: 0);
            m_mlp.activFctIsNonLinear = m_mlp.activFct.IsNonLinear();

            m_mlp.InitStruct(NeuronCount, addBiasColumn: true);

            m_mlp.W[0] = new double[,] {
                {-0.5, -0.78, -0.07},
                {0.54, 0.32, -0.13},
                {-0.29, 0.89, -0.8}};
            m_mlp.W[1] = new double[,] {
                {0.28},
                {-0.94},
                {-0.5},
                {-0.36}};

            m_mlp.Train();

            var expectedOutput = new double[,] {
                {0.99},
                {0.03},
                {0.99},
                {0.03}};

            var output = m_mlp.output.ToString();
            Matrix expectedMatrix = expectedOutput; // double(,) -> Matrix
            var expectedOutputString = expectedMatrix.ToString();
            Assert.AreEqual(output, expectedOutputString);

            var expectedLoss = 0.08;
            float[] targetArray = m_mlp.TargetValue.ToArray();
            var loss = m_mlp.ComputeAverageError(targetArray);
            var lossRounded = Math.Round(loss, 2);
            Assert.AreEqual(expectedLoss, lossRounded);
        }

        [TestMethod()]
        public void VectorizedMatrixMLPXORSinus()
        {
            m_mlp.nbIterations = 200; // Sinus: works
            m_mlp.activFct = new ActivationFunction.SinusFunction();
            m_mlp.lambdaFct = x => m_mlp.activFct.Activation(x, gain: 1, center: 0);
            m_mlp.lambdaFctD = x => m_mlp.activFct.Derivative(x, gain: 1, center: 0);
            m_mlp.activFctIsNonLinear = m_mlp.activFct.IsNonLinear();

            m_mlp.InitStruct(NeuronCount, addBiasColumn: false);

            m_mlp.W[0] = new double[,] {
                {0.7, 0.8},
                {0.04, 0.59}};
            m_mlp.W[1] = new double[,] {
                {0.03},
                {0.66}};

            m_mlp.Train();

            var expectedOutput = new double[,] {
                {0.99},
                {0.0},
                {0.99},
                {0.0}};

            var output = m_mlp.output.ToString();
            Matrix expectedMatrix = expectedOutput; // double(,) -> Matrix
            var expectedOutputString = expectedMatrix.ToString();
            Assert.AreEqual(output, expectedOutputString);

            var expectedLoss = 0.02;
            float[] targetArray = m_mlp.TargetValue.ToArray();
            var loss = m_mlp.ComputeAverageError(targetArray);
            var lossRounded = Math.Round(loss, 2);
            Assert.AreEqual(expectedLoss, lossRounded);
        }

        [TestMethod()]
        public void VectorizedMatrixMLPXORArcTangent()
        {
            m_mlp.nbIterations = 500; // Arc tangent: works
            m_mlp.activFct = new ActivationFunction.ArcTangentFunction();
            m_mlp.lambdaFct = x => m_mlp.activFct.Activation(x, gain: 1, center: 0);
            m_mlp.lambdaFctD = x => m_mlp.activFct.Derivative(x, gain: 1, center: 0);
            m_mlp.activFctIsNonLinear = m_mlp.activFct.IsNonLinear();

            m_mlp.InitStruct(NeuronCount, addBiasColumn: true);

            m_mlp.W[0] = new double[,] {
                {-0.5, -0.78, -0.07},
                {0.54, 0.32, -0.13},
                {-0.29, 0.89, -0.8}};
            m_mlp.W[1] = new double[,] {
                 {0.28},
                {-0.94},
                {-0.5},
                {-0.36}};

            m_mlp.Train();

            var expectedOutput = new double[,] {
                {0.99},
                {0},
                {0.99},
                {0}};

            var output = m_mlp.output.ToString();
            Matrix expectedMatrix = expectedOutput; // double(,) -> Matrix
            var expectedOutputString = expectedMatrix.ToString();
            Assert.AreEqual(output, expectedOutputString);

            var expectedLoss = 0.03;
            float[] targetArray = m_mlp.TargetValue.ToArray();
            var loss = m_mlp.ComputeAverageError(targetArray);
            var lossRounded = Math.Round(loss, 2);
            Assert.AreEqual(expectedLoss, lossRounded);
        }

        [TestMethod()]
        public void VectorizedMatrixMLPXORReLUSigmoid()
        {
            m_mlp.nbIterations = 5000; // ReLUSigmoid: works fine
            m_mlp.activFct = new ActivationFunction.ReLUSigmoidFunction();
            m_mlp.lambdaFct = x => m_mlp.activFct.Activation(x, gain: 1, center: 0);
            m_mlp.lambdaFctD = x => m_mlp.activFct.Derivative(x, gain: 1, center: 0);
            m_mlp.activFctIsNonLinear = m_mlp.activFct.IsNonLinear();

            m_mlp.InitStruct(NeuronCount, addBiasColumn: true);

            m_mlp.W[0] = new double[,] {
                {-0.5, -0.78, -0.07},
                {0.54, 0.32, -0.13},
                {-0.29, 0.89, -0.8}};
            m_mlp.W[1] = new double[,] {
                {0.28},
                {-0.94},
                {-0.5},
                {-0.36}};

            m_mlp.Train();

            var expectedOutput = new double[,] {
                {1.0},
                {0.0},
                {1.0},
                {0.0}};

            var output = m_mlp.output.ToString();
            Matrix expectedMatrix = expectedOutput; // double(,) -> Matrix
            var expectedOutputString = expectedMatrix.ToString();
            Assert.AreEqual(output, expectedOutputString);

            var expectedLoss = 0.01;
            float[] targetArray = m_mlp.TargetValue.ToArray();
            var loss = m_mlp.ComputeAverageError(targetArray);
            var lossRounded = Math.Round(loss, 2);
            Assert.AreEqual(expectedLoss, lossRounded);
        }
    }
}
