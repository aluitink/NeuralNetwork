using System;
using Helvetica.NeuralNetwork.Models;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Helvetica.NeuralNetwork.Test
{
    [TestClass]
    public class SimpleNetworkTrainerTest
    {
        [TestMethod]
        [Ignore]
        public void ConvolutionNetwork()
        {
            int[] layerSizes = new int[5]
            {
                841,
                1014,
                1250,
                100,
                10
            };

            TransferFunction[] transferFunctions = new TransferFunction[5]
            {
                TransferFunction.None,
                TransferFunction.Convolution,
                TransferFunction.Convolution,
                TransferFunction.Linear,
                TransferFunction.Linear
            };

            BackPropagationNetwork backPropagationNetwork = new BackPropagationNetwork(layerSizes, transferFunctions);

            double[] input1 = new double[841];

            for (int i = 0; i < input1.Length; i++)
            {
                if(i % 2 == 0)
                    input1[i] = 1;
            }

            double[] input2 = new double[841];

            for (int i = 0; i < input2.Length; i++)
            {
                input2[i] = 1;
            }

            DataPoint _dp1 = new DataPoint(input1, new[] { 1.0 });

            DataPoint _dp2 = new DataPoint(input2, new[] { 0.0 });

            DataPointCollection _dataPointCollection = new DataPointCollection();
            _dataPointCollection.Add(_dp1);
            _dataPointCollection.Add(_dp2);

            SimpleNetworkTrainer _networkTrainer = new SimpleNetworkTrainer(backPropagationNetwork, _dataPointCollection);

            _networkTrainer.TargetError = 0.0001;
            _networkTrainer.MaxIterations = 1000000;
            _networkTrainer.NudgeScale = 0.8;
            _networkTrainer.NudgeWindow = 100;

            _networkTrainer.TrainNetwork();
            Assert.IsTrue(true, "Never Reached Minimum Error");

            for (int i = _networkTrainer.ErrorHistory.Count - 100; i < _networkTrainer.ErrorHistory.Count; i++)
            {
                Console.WriteLine("{0}: {1:0.00000000}", i, _networkTrainer.ErrorHistory[i]);
            }
        }

        [TestMethod]
        public void CanTrainNetwork()
        {
            //XOR data
            DataPoint _dp1 = new DataPoint(new[] { 1.0, 1.0 }, new[] { 0.0 });
            DataPoint _dp2 = new DataPoint(new[] { 1.0, 0.0 }, new[] { 1.0 });
            DataPoint _dp3 = new DataPoint(new[] { 0.0, 1.0 }, new[] { 1.0 });
            DataPoint _dp4 = new DataPoint(new[] { 0.0, 0.0 }, new[] { 0.0 });

            DataPointCollection _dataPointCollection = new DataPointCollection();
            _dataPointCollection.Add(_dp1);
            _dataPointCollection.Add(_dp2);
            _dataPointCollection.Add(_dp3);
            _dataPointCollection.Add(_dp4);

            int[] _layerSizes = new int[3] { 2, 2, 1 };
            TransferFunction[] _transferFunctions = new TransferFunction[3]
                                                        {
                                                            TransferFunction.None,
                                                            TransferFunction.Sigmoid,
                                                            TransferFunction.Linear
                                                        };

            BackPropagationNetwork _bpn = new BackPropagationNetwork(_layerSizes, _transferFunctions);

            SimpleNetworkTrainer _networkTrainer = new SimpleNetworkTrainer(_bpn, _dataPointCollection);

            _networkTrainer.TargetError = 0.0001;
            _networkTrainer.MaxIterations = 1000000;
            _networkTrainer.NudgeScale = 0.8;
            _networkTrainer.NudgeWindow = 100;

            _networkTrainer.TrainNetwork();
            Assert.IsTrue(true, "Never Reached Minimum Error");

            for (int i = _networkTrainer.ErrorHistory.Count - 100; i < _networkTrainer.ErrorHistory.Count; i++)
            {
                Console.WriteLine("{0}: {1:0.00000000}", i, _networkTrainer.ErrorHistory[i]);
            }
        }
    }
}
