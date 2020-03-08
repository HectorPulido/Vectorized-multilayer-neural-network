using System;
using LinearAlgebra;
using System.Diagnostics; // Debug.WriteLine

namespace VectorizedMultiLayerPerceptron
{
    public class VectorizedMatrixMLP
    {
        public ActivationFunction.IActivationFunction activFct;
        public Func<double, double> lambdaFct;
        public Func<double, double> lambdaFctD;
        public bool activFctIsNonLinear;

        public double LearningRate;
        public int nbIterations;

        public Matrix InputValue;
        public Matrix TargetValue;
        public Matrix output;
        public Matrix LastError;

        public int ExampleCount;

        public Matrix[] W;

        private Random r;
        private Matrix[] error;

        private int LayerCount;
        private int[] NeuronCount;
        private bool addBiasColumn = true;

        public VectorizedMatrixMLP()
        {
        }

        public void InitStruct(int[] aiNeuronCount, bool addBiasColumn)
        {
            this.addBiasColumn = addBiasColumn;
            this.LayerCount = aiNeuronCount.Length;
            //this.NeuronCount = new int[this.LayerCount - 1];
            this.NeuronCount = new int[this.LayerCount];
            for (int i = 0; i < this.LayerCount; i++)
            {
                this.NeuronCount[i] = aiNeuronCount[i];
                if (addBiasColumn && i > 0 && i < this.LayerCount - 1)
                    this.NeuronCount[i] += 1; // Bias
            }
            this.ExampleCount = this.TargetValue.x;
            this.W = new Matrix[this.LayerCount - 1 - 1 + 1];
        }

        public void Randomize()
        {
            this.r = new Random(1);

            for (var i = 0; i <= this.W.Length - 1; i++)
                this.W[i] = Matrix.Random(
                    this.NeuronCount[i] + 1, this.NeuronCount[i + 1], r) * 2 - 1;
        }

        public void Train()
        {
            for (var epoch = 0; epoch <= this.nbIterations - 1; epoch++)
                OneIteration(this.InputValue, this.ExampleCount, epoch,
                    testOnly: false, TargetValue: this.TargetValue, useTargetValue: true);
        }

        public void OneIteration(Matrix InputValue, int ExampleCount, int Iteration,
            bool testOnly, Matrix TargetValue, bool useTargetValue=false)
        {
            this.error = null;
            this.output = null;

            Matrix[] Z = null;
            Matrix[] A = null;
            ForwardPropagation(InputValue, out Z, out A, ExampleCount);

            var maxLayer = LayerCount - 1;
            var maxIndex = A.Length - 1;
            Matrix Zlast = Z[maxLayer];
            // Cut first column for last layer
            var zx = Z[maxLayer].x;
            var zy = Z[maxLayer].y;
            if (addBiasColumn) Zlast = Zlast.Slice(0, 1, zx, zy);

            this.output = A[maxIndex];
            // Cut first column for last index of result matrix
            var ax = A[maxIndex].x;
            var ay = A[maxIndex].y;
            if (addBiasColumn) this.output = this.output.Slice(0, 1, ax, ay);

            this.error = null;
            if (useTargetValue) {
                this.error = new Matrix[this.LayerCount - 1 + 1];
                this.error[this.LayerCount - 1] = this.output - TargetValue;
            }

            if (testOnly) return;

            Matrix[] delta = null;
            BackPropagation(out delta, this.error, Zlast, Z, A);

            GradientDescend(A, delta, this.LearningRate);

            if (Iteration < 10 ||
                   ((Iteration + 1) % 100 == 0 && Iteration < 1000) ||
                   ((Iteration + 1) % 1000 == 0 && Iteration < 10000) ||
                    (Iteration + 1) % 10000 == 0)
            {
                var sMsg = "\n" +
                    "-------" + (Iteration + 1) + "----------------" + "\n" +
                    "Input: " + this.InputValue.ToString() + "\n" +
                    "Output: " + this.output.ToString() + "\n";

                //for (var i = 0; i <= this.LayerCount - 1; i++) {
                //    sMsg += "Error(" + i + ")=" + this.error[i].ToString() + "\n";
                //    sMsg += "A(" + i + ")=" + A[i].ToString() + "\n";
                //}

                var LastError = this.error[this.LayerCount - 1];
                var averageErr = LastError.abs.average * this.ExampleCount;
                sMsg +=
                    "Loss: " + averageErr.ToString("0.000000") + "\n";

                Debug.WriteLine(sMsg);
                Console.WriteLine(sMsg);
            }
        }

        private void ForwardPropagation(Matrix InputValue, out Matrix[] Z, 
            out Matrix[] A, int ExampleCount)
        {
            Z = new Matrix[LayerCount - 1 + 1];
            A = new Matrix[LayerCount - 1 + 1];

            Z[0] = InputValue;
            // Column added with 1 for all examples
            if (addBiasColumn)
                Z[0] = Z[0].AddColumn(Matrix.Ones(ExampleCount, 1));
            A[0] = Z[0];

            for (var i = 1; i <= LayerCount - 1; i++) {
                var AW = A[i - 1] * W[i - 1];

                Z[i] = AW;
                // Column added with 1 for all examples
                if (addBiasColumn)
                    Z[i] = Z[i].AddColumn(Matrix.Ones(ExampleCount, 1));

                A[i] = Matrix.Map(Z[i], this.lambdaFct);
            }

            // How use Relu
            // Change all sigmoid function, for relu function
            // Last A must have no Nonlinear function Matrix, Last A must be Equal To Last Z;
            // because of that Last Delta has not derivated Matrix "Last Delta = Last error Error * 1";
            // The learning rate must be smaller, like 0.001
            // Optionaly you can use a Softmax layer to make a clasifier
            // Use if Relu OR iregularized Values
            if (activFctIsNonLinear) A[A.Length - 1] = Z[Z.Length - 1];
        }

        private void BackPropagation(out Matrix[] delta, Matrix[] error_, 
            Matrix Zlast, Matrix[] Z, Matrix[] A)
        {
            delta = new Matrix[this.LayerCount - 1 + 1];

            // delta(LayerCount - 1) = error_(LayerCount - 1) * sigmoid(Zlast, derivated:=True)
            delta[this.LayerCount - 1] = error_[this.LayerCount - 1] * Matrix.Map(Zlast, this.lambdaFctD);

            for (var i = this.LayerCount - 2; i >= 0; i += -1) {
                var d = delta[i + 1];
                var t = this.W[i].T;
                this.error[i] = d * t;

                // delta(i) = error_(i) * sigmoid(Z(i), derivated:=True)
                delta[i] = this.error[i] * Matrix.Map(Z[i], this.lambdaFctD);

                // Cut first column
                if (addBiasColumn)
                    delta[i] = delta[i].Slice(0, 1, delta[i].x, delta[i].y);
            }
        }

        private void GradientDescend(Matrix[] A, Matrix[] delta, double LearningRate)
        {
            for (var i = 0; i <= W.Length - 1; i++)
                this.W[i] -= A[i].T * delta[i + 1] * this.LearningRate;
        }

        public float ComputeAverageError()
        {
            this.LastError = this.error[this.LayerCount - 1];
            var averageErr = Convert.ToSingle(this.LastError.abs.average * this.ExampleCount);
            return averageErr;
        }

        public float ComputeAverageError(float[] targets_array)
        {
            this.LastError = this.error[this.LayerCount - 1];
            float averageErr = Convert.ToSingle(this.LastError.abs.average * this.ExampleCount);
            return averageErr;
        }
    }

}
