using System;

namespace ActivationFunction
{
    /// <summary>
    /// Interface for all activation functions
    /// </summary>
    public interface IActivationFunction
    {

        /// <summary>
        /// Activation function
        /// </summary>
        double Activation(double x, double gain, double center);

        double Derivative(double x, double gain, double center);

        /// <summary>
        /// Return true if the activation function is non linear,
        ///  for example Rectified Linear Unit (ReLU)
        /// </summary>
        bool IsNonLinear();
    }

    /// <summary>
    /// Implements f(x) = Sigmoid
    /// </summary>
    public class SigmoidFunction : IActivationFunction
    {

        public double Activation(double x, double gain, double center)
        {
            return SigmoidFunction.CommonActivation(x, gain, center);
        }

        public static double CommonActivation(double x, double gain, double center)
        {
            double y = 1 / (1 + Math.Exp(-gain * (x - center)));
            return y;
        }

        public double Derivative(double x, double gain, double center)
        {
            return SigmoidFunction.CommonDerivate(x, gain, center);
        }

        public static double CommonDerivate(double x, double gain, double center)
        {
            double y;
            if (gain == 1)
            {
                double fx = SigmoidFunction.CommonActivation(x, gain, center);
                y = fx * (1 - fx);
            }
            else
            {
                double xc = x - center;
                double exp = Math.Exp(-gain * xc);
                double expP1 = 1 + exp;
                y = gain  * exp / (expP1 * expP1);
            }
            return y;
        }

        public bool IsNonLinear()
        {
            return false;
        }
    }

    /// <summary>
    /// Implements f(x) = Hyperbolic Tangent
    /// </summary>
    public class HyperbolicTangentFunction : IActivationFunction
    {

        public double Activation(double x, double gain, double center)
        {
            return HyperbolicTangentFunction.CommonActivation(x, gain, center);
        }

        public static double CommonActivation(double x, double gain, double center)
        {
            double xc = x - center;
            double y = 2 / (1 + Math.Exp(-2 * xc)) - 1;
            return y;
        }

        public double Derivative(double x, double gain, double center)
        {
            double y;
            if (gain == 1)
            {
                double fx = HyperbolicTangentFunction.CommonActivation(x, gain, center);
                y = 1 - (fx * fx);
            }
            else
            {
                double xc = x - center;
                double exp = Math.Exp(-2 * (gain * xc));
                double expP1 = 1 + exp;
                y = 4 * gain * exp / (expP1 * expP1);
            }

            return y;
        }

        public bool IsNonLinear()
        {
            return false;
        }
    }

    /// <summary>
    /// Implements f(x) = Exponential Linear Unit (ELU)
    /// </summary>
    public class ELUFunction : IActivationFunction
    {

        public double Activation(double x, double gain, double center)
        {
            double xc = x - center;
            double y;
            if (xc >= 0)
                y = xc;
            else
                y = gain * (Math.Exp(xc) - 1);
            return y;
        }

        public double Derivative(double x, double gain, double center)
        {
            if (gain < 0) return 0;

            double xc = x - center;
            double y;
            if (xc >= 0)
                y = 1;
            else
                //y = x + gain;
                y = gain + Activation(x, gain, center);
            return y;
        }

        public bool IsNonLinear()
        {
            return false;
        }
    }

    /// <summary>
    /// Implements Rectified Linear Unit (ReLU)
    /// </summary>
    public class ReLUFunction : IActivationFunction
    {

        public double Activation(double x, double gain, double center)
        {
            double xc = x - center;
            return Math.Max(xc * gain, 0);
        }

        public double Derivative(double x, double gain, double center)
        {
            if (x >= center) return gain;
            return 0;
        }

        public bool IsNonLinear()
        {
            return true;
        }
    }

    /// <summary>
    /// Implements Rectified Linear Unit (ReLU) with sigmoid for derivate
    /// </summary>
    public class ReLUSigmoidFunction : IActivationFunction
    {

        public double Activation(double x, double gain, double center)
        {
            double xc = x - center;
            return Math.Max(xc * gain, 0);
        }

        public double Derivative(double x, double gain, double center)
        {
            return SigmoidFunction.CommonDerivate(x, gain, center); // Sigmoid derivative
        }

        public bool IsNonLinear()
        {
            return false;
        }
    }

    /// <summary>
    /// Implements f(x) = Gaussian
    /// </summary>
    public class GaussianFunction : IActivationFunction
    {

        public double Activation(double x, double gain, double center)
        {
            double xc = x - center;
            double xg = -gain * (xc * xc);
            double y = Math.Exp(xg);
            return y;
        }

        public double Derivative(double x, double gain, double center)
        {
            double xc = x - center;
            double g2 = gain * gain;
            double exp = Math.Exp(-g2 * xc * xc);
            double y = -2 * g2 * xc * exp;
            return y;
        }

        public bool IsNonLinear()
        {
            return false;
        }
    }

    /// <summary>
    /// Implements f(x) = sin(x)
    /// </summary>
    public class SinusFunction : IActivationFunction
    {

        public double Activation(double x, double gain, double center)
        {
            double y = gain * Math.Sin(x - center);
            return y;
        }

        public double Derivative(double x, double gain, double center)
        {
            double y = gain * Math.Cos(x - center);
            return y;
        }

        public bool IsNonLinear()
        {
            return false;
        }
    }

    /// <summary>
    /// Implements f(x) = ArcTangent(x)
    /// </summary>
    public class ArcTangentFunction : IActivationFunction
    {

        public double Activation(double x, double gain, double center)
        {
            double y = gain * Math.Atan(gain * (x - center));
            return y;
        }

        public double Derivative(double x, double gain, double center)
        {
            double xc = x - center;
            // https://www.wolframalpha.com/input/?i=arctan(alpha+*+x)+derivative
            double y = gain / (1 + gain * gain * xc * xc);
            return y;
        }

        public bool IsNonLinear()
        {
            return false;
        }
    }
}