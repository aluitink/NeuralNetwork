using Helvetica.NeuralNetwork.Models;

namespace Helvetica.NeuralNetwork.Interfaces
{
    public interface INeuralNetworkTrainer
    {
        double Error { get; }
        double MaxError { get; set; }
        double MaxIterations { get; set; }
        double TrainingRate { get; set; }
        double Momentum { get; set; }
        DataPointCollection DataSet { get; set; }
        INeuralNetwork Network { get; set; }

        bool TrainNetwork();
    }
}
