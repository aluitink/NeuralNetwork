using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Helvetica.NeuralNetwork.Models
{
    public class NetworkSettingsBase
    {
        public String NetworkName { get; set; }
        public int LayerCount { get; set; }
        public int InputSize { get; set; }
        public int OutputSize { get; set; }
        public int[] LayerSizes { get; set; }
        public TransferFunction[] TransferFunctions { get; set; }
        public double[][] LayerInput { get; set; }
        public double[][] LayerOutput { get; set; }
        public double[][] Bias { get; set; }
        public double[][] Delta { get; set; }
        public double[][] PreviousBiasDelta { get; set; }
        public double[][][] Weight { get; set; }
        public double[][][] PreviousWeightDelta { get; set; }

    }
}
