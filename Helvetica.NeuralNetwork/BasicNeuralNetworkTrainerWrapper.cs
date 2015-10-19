using Helvetica.NeuralNetwork.Interfaces;
using Helvetica.NeuralNetwork.Models;

namespace Helvetica.NeuralNetwork
{
    public class BasicNeuralNetworkTrainerWrapper: INeuralNetworkTrainer
    {
        public double Error { get; private set; }
        public double MaxError { get; set; }
        public double MaxIterations { get; set; }
        public double TrainingRate { get; set; }
        public double Momentum { get; set; }
        public DataPointCollection DataSet { get; set; }
        public INeuralNetwork Network { get; set; }

        private SimpleNetworkTrainer _networkTrainer;
        private BackPropagationNetwork _network;
        public BasicNeuralNetworkTrainerWrapper(DataPointCollection dataSet)
        {
            DataSet = dataSet;

            int[] layerSizes = new int[10] { DataSet.DataPointBound, 5, 5, 5, 5, 5, 5, 5, 3, 1 };
            TransferFunction[] transferFunctions = new TransferFunction[10]
                                                        {
                                                            TransferFunction.None,
                                                            TransferFunction.RationalSigmoid,
                                                            TransferFunction.RationalSigmoid,
                                                            TransferFunction.Sigmoid,
                                                            TransferFunction.Sigmoid,
                                                            TransferFunction.Sigmoid,
                                                            TransferFunction.Sigmoid,
                                                            TransferFunction.Gaussian,
                                                            TransferFunction.Gaussian,
                                                            TransferFunction.Linear
                                                        };


            Network.Initialize(layerSizes, transferFunctions);
            _network = new BackPropagationNetwork(layerSizes, transferFunctions);
            _networkTrainer = new SimpleNetworkTrainer(_network, DataSet);
        }

        public bool TrainNetwork()
        {
            _networkTrainer.TargetError = 0.0001;
            _networkTrainer.MaxIterations = 100000000;
            _networkTrainer.NudgeScale = .8;
            _networkTrainer.NudgeWindow = 100;

            _networkTrainer.TrainNetwork();
            return true;
        }
    }
}
