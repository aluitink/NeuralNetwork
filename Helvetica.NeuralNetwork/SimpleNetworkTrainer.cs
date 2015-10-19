using Helvetica.NeuralNetwork.Models;

namespace Helvetica.NeuralNetwork
{
    public class SimpleNetworkTrainer: NetworkTrainer
    {
        public SimpleNetworkTrainer(BackPropagationNetwork network, DataPointCollection dataSet): base(network, dataSet)
        {
            
        }

        protected override void BeforeTrainEpoch()
        {
            base.BeforeTrainEpoch();
        }

        protected override void AfterTrainEpoch()
        {
            SmallestError = ErrorHistory[ErrorHistory.Count - 1] < SmallestError
                                ? ErrorHistory[ErrorHistory.Count - 1]
                                : SmallestError;
            base.AfterTrainEpoch();
        }

        protected override void BeforeTrainDataPoint(ref double[] input, ref double[] output, int index)
        {
            base.BeforeTrainDataPoint(ref input, ref output, index);
        }

        protected override void AfterTrainDataPoint(ref double[] input, ref double[] output, int index)
        {
            base.AfterTrainDataPoint(ref input, ref output, index);
        }
    }
}
