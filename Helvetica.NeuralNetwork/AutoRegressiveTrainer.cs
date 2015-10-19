using System;
using System.Diagnostics;
using Helvetica.NeuralNetwork.Models;

namespace Helvetica.NeuralNetwork
{
    public class AutoRegressiveTrainer : NetworkTrainer
    {
        Stopwatch stopwatch = new Stopwatch();
        private double[] lastOutput;
        private double errorDifference = 0;
        private string errorTrend;
        public AutoRegressiveTrainer(BackPropagationNetwork network, DataPointCollection dataSet)
            : base(network, dataSet)
        {
            lastOutput = new double[dataSet.Count];
        }

        protected override void BeforeTrainEpoch()
        {
            if (CurrentIteration % 500 == 0)
            {
                stopwatch.Start();
                Console.Clear();
                Console.WriteLine("Start Iteration: {0}", CurrentIteration);
            }
            base.BeforeTrainEpoch();
        }

        protected override void AfterTrainEpoch()
        {
            stopwatch.Stop();

            if (errorDifference > 0)
                errorTrend = "";
            else
                errorTrend = "GOOD";
            if (CurrentIteration % 500 == 0)
            {
                
                Console.WriteLine("Error {0}", ErrorHistory[ErrorHistory.Count - 1]);
                Console.WriteLine("ErrorChange: {0}", errorDifference);
                Console.WriteLine("ErrorTrend: {0}", errorTrend);
                Console.WriteLine("Time: {0}", stopwatch.ElapsedTicks);
            }
            
            stopwatch.Reset();
            base.AfterTrainEpoch();
        }

        protected override void BeforeTrainDataPoint(ref double[] input, ref double[] output, int index)
        {
            //if (CurrentIteration % 100 == 0)
            //{
            //    foreach (double d in input)
            //    {
            //        Console.Write("[{1}]:{0} ", d, index);
            //    }

            //    Console.WriteLine();
            //}

            //if (CurrentIteration > 2)
                //input[4] = lastOutput[index];
            base.BeforeTrainDataPoint(ref input, ref output, index);
        }

        protected override void AfterTrainDataPoint(ref double[] input, ref double[] output, int index)
        {
            //if (CurrentIteration > 2)
            //{
            //    double[] opt = new double[1];
            //    Network.Run(ref input, out opt);
            //    lastOutput[index] = opt[0];
            //}

            if(ErrorHistory.Count > 2)
                errorDifference = ErrorHistory[ErrorHistory.Count - 2] - ErrorHistory[ErrorHistory.Count - 1];
            base.AfterTrainDataPoint(ref input, ref output, index);
        }
    }
}
