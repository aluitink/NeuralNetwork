using System;

namespace Helvetica.NeuralNetwork.Models
{
    [Serializable]
    public class DataPoint
    {
        public int HashCode
        {
            get
            {
                if(Input != null && Output != null)
                {
                    int h1 = 0;
                    int h2 = 0;
                    foreach (double d in Input)
                    {
                        h1 =+ d.GetHashCode();
                    }
                    foreach (double d in Output)
                    {
                        h2 = +d.GetHashCode();
                    }
                    return (h1) / (h2);
                }
                return 0;
            }
        }

        public double[] Input;
        public double[] Output;

        public DataPoint(){}
        public DataPoint(double[] input, double[] output)
        {
            Input = (double[])input.Clone();
            Output = (double[])output.Clone();
        }
    }
}
