using System;
using System.Text; // StringBuilder 

namespace LinearAlgebra
{
    public struct Matrix
    {
        double[,] _matrix;
        public double[,] ToMatrix { get { return (double[,])_matrix.Clone(); } set { _matrix = value; } }
        public int X { get { return _matrix.GetLength(0); } }
        public int Y { get { return _matrix.GetLength(1); } }
        public Matrix T { get { return Transpose(this); } }
        public Matrix size { get { return (new double[,] { { X , Y } } ); } }
        public Matrix abs { get { return Absolute(this); } }
        public double average { get { return Average(this); } }

        //constructor
        public Matrix(int sizex, int sizey)
        {
            _matrix = new double[sizex, sizey];
        }
        public Matrix(double[,] matrix)
        {
            _matrix = matrix;
        }
        //Setup
        public static implicit operator Matrix(double[,] matrix)
        {
            return new Matrix(matrix);
        }
        public static implicit operator double[,](Matrix matrix)
        {
            return matrix.ToMatrix;
        }
        //values
        public void SetValue(int x, int y, double value)
        {
            if (_matrix == null)
                throw new ArgumentException("Matrix can not be null");
            _matrix[x, y] = value;
        }
        public double GetValue(int x, int y)
        {
            if (_matrix == null)
                throw new ArgumentException("Matrix can not be null");
            return _matrix[x, y];
        }
        public Matrix Slice(int x1, int y1, int x2, int y2)
        {
            if (_matrix == null)
                throw new ArgumentException("Matrix can not be null");

            if (x1 >= x2 || y1 >= y2 || x1 < 0 || x2 < 0 || y1 < 0 || y2 < 0)
                throw new ArgumentException("Dimensions are not valid");

            double[,] slice = new double[x2 - x1, y2 - y1];

            for (int i = x1; i < x2; i++)
            {
                for (int j = y1; j < y2; j++)
                {
                    slice[i - x1, j - y1] = _matrix[i, j];
                }
            }
            return (slice);
        }
        public Matrix Slice(int x, int y)
        {
            return Slice(0, 0, x, y);
        }
        public Matrix GetRow(int x)
        {
            if (_matrix == null)
                throw new ArgumentException("Matrix can not be null");

            double[,] row = new double[1, Y];
            for (int j = 0; j < Y; j++)
            {
                row[0, j] = _matrix[x, j];
            }
            return (row);
        }
        public Matrix GetColumn(int y)
        {
            if (_matrix == null)
                throw new ArgumentException("Matrix can not be null");

            double[,] column = new double[X, 1];
            for (int i = 0; i < X; i++)
            {
                column[i, 0] = _matrix[i, y];
            }
            return (column);
        }
        public Matrix AddColumn(Matrix m2)
        {
            if (_matrix == null)
                throw new ArgumentException("Matrix can not be null");
            if (m2.Y != 1 || m2.X != X)
                throw new ArgumentException("Invalid dimensions");

            double[,] newMatrix = new double[X, Y + 1];
            double[,] m = _matrix;

            for (int i = 0; i < X; i++)
            {
                newMatrix[i, 0] = m2.GetValue(i, 0);
            }
            MatrixLoop((i, j) =>
            {
                newMatrix[i, j + 1] = m[i, j];
            }, X, Y);
            return (newMatrix);
        }
        public Matrix AddRow(Matrix m2)
        {
            if (_matrix == null)
                throw new ArgumentException("Matrix can not be null");
            if (m2.X != 1 || m2.Y != Y)
                throw new ArgumentException("Invalid dimensions");

            double[,] newMatrix = new double[X + 1, Y];
            //double[,] m = matrix;
            double[,] m = _matrix;

            for (int j = 0; j < Y; j++)
            {
                newMatrix[0, j] = m2.GetValue(0, j);
            }
            MatrixLoop((i, j) =>
            {
                newMatrix[i + 1, j] = m[i, j];
            }, X, Y);
            return (newMatrix);
        }
        //Overriding
        public override string ToString()
        {
            return ToString();
        }
        public string ToString(string dec = "0.00")
        {

            //string c = "{";
            //for (int i = 0; i < x; i++)
            //{
            //    c += "{";
            //    for (int j = 0; j < y; j++)
            //    {
            //        c += matrix[i, j].ToString(dec) + ", ";
            //    }
            //    c += "},\n";
            //}
            //c += "}";
            //return c;

            var sb = new StringBuilder();
            sb.AppendLine("{");
            for (int i = 0; i < X; i++)
            {
                sb.Append(" {");
                for (int j = 0; j < Y; j++)
                {
                    string val = _matrix[i, j].ToString(dec).Replace(",", ".");
                    sb.Append(val);
                    if (j < Y-1) sb.Append(", ");
                }
                sb.Append("}");
                if (i < X-1) sb.Append("," + Environment.NewLine);
            }
            sb.Append("}");
            return sb.ToString();
        }
        //PREMADES
        public static Matrix Zeros(int x, int y)
        {
            double[,] zeros = new double[x, y];
            MatrixLoop((i, j) => {
                zeros[i, j] = 0;
            }, x, y);
            return (zeros);
        }
        public static Matrix Ones(int x, int y)
        {
            double[,] ones = new double[x, y];
            MatrixLoop((i, j) => {
                ones[i, j] = 1;
            }, x, y);
            return (ones);
        }
        public static Matrix Identy(int x)
        {
            double[,] identy = new double[x, x];
            MatrixLoop((i, j) => {
                if (i == j)
                    identy[i, j] = 1;
                else
                    identy[i, j] = 0;
            }, x, x);
            return (identy);
        }
        public static Matrix Random(int x, int y, Random r)
        {
            double[,] random = new double[x, y];
            MatrixLoop((i, j) => {
                random[i, j] = r.NextDouble();
            }, x, y);
            return (random);
        }
        //Operations
        //Transpose
        public static Matrix Transpose(Matrix m)
        {
            double[,] mT = new double[m.Y, m.X];
            MatrixLoop((i, j) => {
                mT[j, i] = m.GetValue(i, j);
            }, m.X, m.Y);
            return (mT);
        }
        //ADDITIONS
        public static Matrix operator +(Matrix m1, Matrix m2)
        {
            return MatSum(m1, m2);
        }
        public static Matrix operator +(Matrix m2, double m1)
        {
            return MatdoubleSum(m1, m2);
        }
        public static Matrix MatdoubleSum(double m1, Matrix m2)
        {
            double[,] a = m2;
            double[,] b = new double[m2.X, m2.Y];

            MatrixLoop((i, j) => {

                b[i, j] = a[i, j] + m1;

            }, b.GetLength(0), b.GetLength(1));

            return (b);
        }
        public static Matrix MatSum(Matrix m1, Matrix m2, bool neg = false)
        {
            if (m1.X != m2.X || m1.Y != m2.Y)
                throw new ArgumentException("Matrix must have the same dimensions");

            double[,] a = m1;
            double[,] b = m2;
            double[,] c = new double[m1.X, m2.Y];
            MatrixLoop((i, j) => {
                if (!neg)
                    c[i, j] = a[i, j] + b[i, j];
                else
                    c[i, j] = a[i, j] - b[i, j];
            }, c.GetLength(0), c.GetLength(1));
            return (c);
        }
        //SUBSTRACTIONS
        public static Matrix operator -(Matrix m1, Matrix m2)
        {
            return MatSum(m1, m2, true);
        }
        public static Matrix operator -(Matrix m2, double m1)
        {
            return MatdoubleSum(-m1, m2);
        }
        //MULTIPLICATIONS
        public static Matrix operator *(Matrix m2, double m1)
        {
            return MatdoubleMult(m2, m1);
        }
        public static Matrix operator *(Matrix m1, Matrix m2)
        {
            if (m1.X == m2.X && m1.Y == m2.Y)
                return DeltaMult(m1, m2);
            return MatMult(m1, m2);
        }
        public static Matrix MatdoubleMult(Matrix m2, double m1)
        {
            double[,] a = m2;
            double[,] b = new double[m2.X, m2.Y];

            MatrixLoop((i, j) => {

                b[i, j] = a[i, j] * m1;

            }, b.GetLength(0), b.GetLength(1));

            return (b);
        }
        public static Matrix MatMult(Matrix m1, Matrix m2)
        {
            if (m1.Y != m2.X)
                throw new ArgumentException("Matrix must have compatible dimensions");
            int n = m1.X;
            int m = m1.Y;
            int p = m2.Y;

            double[,] a = m1;
            double[,] b = m2;
            double[,] c = new double[n, p];
            MatrixLoop((i, j) => {
                double sum = 0;
                for (int k = 0; k < m; k++)
                {
                    sum += a[i, k] * b[k, j];
                }
                c[i, j] = sum;

            }, n, p);
            return (c);
        }
        public static Matrix DeltaMult(Matrix m1, Matrix m2)
        {
            if (m1.X != m2.X || m1.Y != m2.Y)
                throw new ArgumentException("Matrix must have the same dimensions");
            double[,] output = new double[m1.X, m2.Y];
            MatrixLoop((i, j) =>
            {
                output[i, j] = m1.GetValue(i, j) * m2.GetValue(i, j);
            }, m1.X, m2.Y);
            return (output);
        }
        //DIVISION
        public static Matrix operator / (Matrix m2, double m1)
        {
            return MatdoubleMult(m2, 1 / m1);
        }
        //POW
        public static Matrix operator ^(Matrix m2, double m1)
        {
            return Pow(m2, m1);
        }
        public static Matrix Pow(Matrix m2, double m1)
        {
            double[,] output = new double[m2.X, m2.Y];
            MatrixLoop((i, j) => {
                output[i, j] = Math.Pow(m2.GetValue(i, j), m1);
            }, m2.X, m2.Y);
            return (output);
        }
        public Matrix Pow(double m1)
        {
            return Pow(this, m1);
        }
        //Sumatory 
        public static Matrix Sumatory(Matrix m, AxisZero dimension = AxisZero.none)
        {
            double[,] output;
            if (dimension == AxisZero.none)
                output = new double[1, 1];
            else if (dimension == AxisZero.horizontal)
                output = new double[m.X, 1];
            else if (dimension == AxisZero.vertical)
                output = new double[1, m.Y];
            else
                throw new ArgumentException("The dimension must be -1, 0 or 1");

            if (dimension == AxisZero.none)
            {
                MatrixLoop((i, j) =>
                {
                    output[0, 0] += m.GetValue(i, j);
                }, m.X, m.Y);
            }
            else if (dimension == AxisZero.horizontal)
            {
                MatrixLoop((i, j) =>
                {
                    output[i, 0] += m.GetValue(i, j);
                }, m.X, m.Y);
            }
            else if (dimension == AxisZero.vertical)
            {
                MatrixLoop((i, j) =>
                {
                    output[0, j] += m.GetValue(i, j);
                }, m.X, m.Y);
            }
            return (output);
        }
        public Matrix Sumatory(AxisZero dimension = AxisZero.none)
        {
            return Sumatory(this, dimension);
        }
        //DOT PRODUCT
        public Matrix Dot(Matrix m2)
        {
            return Dot(this, m2);
        }
        public static Matrix Dot(Matrix m1, Matrix m2)
        {
            return m1 * m2.T;
        }
        //ABS
        public static Matrix Absolute(Matrix m)
        {
            double[,] d = m;
            MatrixLoop((i, j) => { d[i, j] = Math.Abs(m.GetValue(i, j)); }, m.X, m.Y);
            return (d);
        }
        public static double Average(Matrix m)
        {
            double d = 0;
            MatrixLoop((i, j) => { d += m.GetValue(i, j); }, m.X, m.Y);
            return d / (m.X * m.Y);
        }
        //Handlers
        public static void MatrixLoop(Action<int, int> e, int x, int y)
        {
            for (int i = 0; i < x; i++)
            {
                for (int j = 0; j < y; j++)
                {
                    e(i, j);
                }
            }
        }

        /// <summary>
        /// Apply a function to each element of the array
        /// </summary>
        /// <param name="m"></param>
        /// <param name="lambdaFct"></param>
        /// <returns></returns>
        public static Matrix Map(Matrix m, Func<double, double> lambdaFct)
        {
            Matrix c = new Matrix(m.X, m.Y);
            for (int i = 0; i < m.X; i++)
            {
                for (int j = 0; j < m.Y; j++)
                {
                    c._matrix[i, j] = lambdaFct.Invoke(m._matrix[i, j]);
                }
            }
            return c;
        }

        /// <summary>
        /// Convert whole Matrix object to array
        /// </summary>
        public float[] ToArray()
        {
            //float[] array = new float[_matrix.Length - 1];
            float[] array = new float[_matrix.Length] ;
            int k = 0;
            for (int i = 0; i<=this.X - 1; i++)
            {
                for (int j = 0; j<=this.Y - 1; j++)
                {
                    array[k++] = Convert.ToSingle(_matrix[i, j]);
                }
            }
            return array;
        }
    }
}
