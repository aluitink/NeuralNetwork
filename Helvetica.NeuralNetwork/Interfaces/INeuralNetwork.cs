using Helvetica.NeuralNetwork.Models;

namespace Helvetica.NeuralNetwork.Interfaces
{
    public interface INeuralNetwork
    {
        void Initialize(int[] layerSizes, TransferFunction[] layerFunctions);
        void Execute(ref double[] input, out double[] output);
        double Train(ref double[] input, ref double[] output, double trainingRate, double momentum);
    }
}
