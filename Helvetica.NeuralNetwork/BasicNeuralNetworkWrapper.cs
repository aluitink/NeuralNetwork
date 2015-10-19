using Helvetica.NeuralNetwork.Interfaces;
using Helvetica.NeuralNetwork.Models;

namespace Helvetica.NeuralNetwork
{
    public class BasicNeuralNetworkWrapper: INeuralNetwork
    {
        private BackPropagationNetwork _network;

        public void Initialize(int[] layerSizes, TransferFunction[] layerFunctions)
        {
            _network = new BackPropagationNetwork(layerSizes, layerFunctions);
        }

        public double Train(ref double[] input, ref double[] output, double trainingRate, double momentum)
        {
            return _network.Train(ref input, ref output, trainingRate, momentum);
        }

        public void Execute(ref double[] input, out double[] output)
        {
            _network.Run(ref input, out output);
        }
    }
}
