using System;
using System.IO;
using System.Xml;
using Helvetica.NeuralNetwork.Models;

namespace Helvetica.NeuralNetwork
{
    public static class Gaussian
    {
        private static readonly Random _random = new Random();
        public static double GetRandomGaussian()
        {
            return GetRandomGaussian(0, 1.0);
        }
        public static double GetRandomGaussian(double mean, double standardDeviation)
        {
            double rVal1, rVal2;
            GetRandomGaussian(mean, standardDeviation, out rVal1, out rVal2);
            return rVal1;
        }
        public static void GetRandomGaussian(double mean, double standardDeviation, out double val1, out double val2)
        {
            double u, v, s, t;

            do
            {
                u = 2 * _random.NextDouble() - 1;
                v = 2 * _random.NextDouble() - 1;
            } while (u*u + v*v > 1 || (u == 0 && v == 0));

            s = u * u + v * v;
            t = Math.Sqrt((-2.0 * Math.Log(s))/s);

            val1 = standardDeviation * u * t + mean;
            val2 = standardDeviation * v * t + mean;
        }
    }

    public class BackPropagationNetwork
    {
        #region Public

        public String NetworkName { get; set; }

        #endregion

        #region Private
        private int _layerCount;
        private int _inputSize;
        private int[] _layerSize;
        private TransferFunction[] _transferFunction;

        //[x][y] x = layer y = node
        private double[][] _layerInput;
        private double[][] _layerOutput;
        private double[][] _bias;
        private double[][] _delta;
        //bias for the delta of the previous training epoch
        private double[][] _previousBiasDelta;

        private double[][][] _weight;
        private double[][][] _previousWeightDelta;

        private XmlDocument _networkDefinition;
        private int[] _initLayerSizes;
        private TransferFunction[] _initTransferFunctions;
        #endregion

        #region Construction
        
        public BackPropagationNetwork(int[] initLayerSizes, TransferFunction[] initTransferFunctions)
        {
            //Validate the input data
            if (initTransferFunctions.Length != initLayerSizes.Length || initTransferFunctions[0] != TransferFunction.None)
                throw new ArgumentException("Invalid parameters");

            NetworkName = "Default";

            _initLayerSizes = initLayerSizes;
            _initTransferFunctions = initTransferFunctions;
            Initialize();
            //Initialize weights
            RandomizeWeights();
        }

        public BackPropagationNetwork(String filePath)
        {
            
            Load(filePath);
        }

        #endregion

        #region Public Methods
        
        public void Initialize()
        {
            //Initialize network layers
            _layerCount = _initLayerSizes.Length - 1;
            _inputSize = _initLayerSizes[0];
            _layerSize = new int[_layerCount];

            for (int i = 0; i < _layerCount; i++)
                _layerSize[i] = _initLayerSizes[i + 1];

            _transferFunction = new TransferFunction[_layerCount];
            for (int i = 0; i < _layerCount; i++)
                _transferFunction[i] = _initTransferFunctions[i + 1];

            _layerInput = new double[_layerCount][];
            _layerOutput = new double[_layerCount][];

            _delta = new double[_layerCount][];
            _bias = new double[_layerCount][];
            _previousBiasDelta = new double[_layerCount][];

            _weight = new double[_layerCount][][];
            _previousWeightDelta = new double[_layerCount][][];


            for (int l = 0; l < _layerCount; l++)
            {
                //Fill 2 dimensional array
                _layerInput[l] = new double[_layerSize[l]];
                _layerOutput[l] = new double[_layerSize[l]];

                _delta[l] = new double[_layerSize[l]];
                _bias[l] = new double[_layerSize[l]];
                _previousBiasDelta[l] = new double[_layerSize[l]];

                //Fill 3 dimensional array
                _weight[l] = new double[l == 0 ? _inputSize : _layerSize[l - 1]][];
                _previousWeightDelta[l] = new double[l == 0 ? _inputSize : _layerSize[l - 1]][];

                for (int i = 0; i < (l == 0 ? _inputSize : _layerSize[l - 1]); i++)
                {
                    _weight[l][i] = new double[_layerSize[l]];
                    _previousWeightDelta[l][i] = new double[_layerSize[l]];
                }
            }
        }

        public void Run(ref double[] input, out double[] output)
        {
            //Make sure we have enough data
            if (input.Length != _inputSize)
                throw new ArgumentException("Input data is not of the correct dimention");

            // Dimension
            output = new double[_layerSize[_layerCount - 1]];

            //Run the network

            for (int l = 0; l < _layerCount; l++)
            {
                for (int j = 0; j < _layerSize[l]; j++)
                {
                    double sum = 0.0;
                    for (int i = 0; i < (l == 0 ? _inputSize : _layerSize[l - 1]); i++)
                        sum += _weight[l][i][j] * (l == 0 ? input[i] : _layerOutput[l - 1][i]);

                    sum += _bias[l][j];
                    _layerInput[l][j] = sum;

                    _layerOutput[l][j] = TransferFunctions.Evaluate(_transferFunction[l], sum);
                }
            }

            //Copy the output to the output array
            for (int i = 0; i < _layerSize[_layerCount - 1]; i++)
                output[i] = _layerOutput[_layerCount - 1][i];
        }

        public double Train(ref double[] input, ref double[] desired, double trainingRate, double momentum)
        {
            // Parameter Validation
            if (input.Length != _inputSize)
                throw new ArgumentException("Invalid input parameter", "input");
            if (desired.Length != _layerSize[_layerCount - 1])
                throw new ArgumentException("Invalid input parameter", "desired");

            //Locals
            double error = 0.0, sum = 0.0, weightDelta = 0.0, biasDelta = 0.0;
            double[] output = new double[_layerSize[_layerCount - 1]];


            //Run the network
            Run(ref input, out output);

            for (int l = _layerCount - 1; l >= 0; l--)
            {
                //Output layer
                if(l == _layerCount - 1)
                {
                    for (int k = 0; k < _layerSize[l]; k++)
                    {
                        _delta[l][k] = output[k] - desired[k];
                        error += Math.Pow(_delta[l][k], 2);
                        _delta[l][k] *= TransferFunctions.EvaluateDerivative(_transferFunction[l], _layerInput[l][k]);
                    }
                }
                else //Hidden layer
                {
                    for (int i = 0; i < _layerSize[l]; i++)
                    {
                        sum = 0.0;
                        for (int j = 0; j < _layerSize[l + 1]; j++)
                        {
                            sum += _weight[l + 1][i][j]*_delta[l + 1][j];
                        }
                        sum += TransferFunctions.EvaluateDerivative(_transferFunction[l], _layerInput[l][i]);

                        _delta[l][i] = sum;
                    }
                }
            }

            // Update the weights
            for (int l = 0; l < _layerCount; l++)
            {
                for (int i = 0; i < (l == 0 ? _inputSize : _layerSize[l - 1]); i++)
                {
                    for (int j = 0; j < _layerSize[l]; j++)
                    {
                        weightDelta = trainingRate * _delta[l][j] * (l == 0 ? input[i] : _layerOutput[l - 1][i])
                                    + momentum * _previousWeightDelta[l][i][j];
                        _weight[l][i][j] -= weightDelta;
                        //replace
                        _previousWeightDelta[l][i][j] = weightDelta;
                    }
                }
            }

            // Update the biases
            for (int l = 0; l < _layerCount; l++)
            {
                for (int i = 0; i < _layerSize[l]; i++)
                {
                    biasDelta = trainingRate * _delta[l][i];
                    _bias[l][i] -= biasDelta + momentum * _previousBiasDelta[l][i];
                    //replace
                    _previousBiasDelta[l][i] = biasDelta;
                }
            }

            return error;
        }

        public void Nudge(double scalar)
        {
            for (int l = 0; l < _layerCount; l++)
            {
                for (int j = 0; j < _layerSize[l]; j++)
                {
                    // Nudge weights
                    for (int i = 0; i < (l == 0 ? _inputSize : _layerSize[l - 1]); i++)
                    {
                        double w = _weight[l][i][j];
                        double u = Gaussian.GetRandomGaussian(0.0, w * scalar);

                        _weight[l][i][j] += u;
                        _previousWeightDelta[l][i][j] = 0.0;
                    }

                    // Nudge bias

                    double b = _bias[l][j];
                    double v = Gaussian.GetRandomGaussian(0.0, b * scalar);
                    _bias[l][j] += v;
                    _previousBiasDelta[l][j] = 0.0;
                }
            }
        }

        public void Save(string filePath)
        {
            if (filePath == null)
                throw new ArgumentException("filePath");

            XmlWriterSettings settings = new XmlWriterSettings();
            settings.Indent = true;

            XmlWriter writer = XmlWriter.Create(filePath, settings);
            // Begin document

            writer.WriteStartElement("NeuralNetwork");
            writer.WriteAttributeString("Type", "BackPropagation");

            // Parameters element
            writer.WriteStartElement("Parameters");

            writer.WriteElementString("NetworkName", NetworkName);
            writer.WriteElementString("InputSize", _inputSize.ToString());
            writer.WriteElementString("LayerCount", _layerCount.ToString());

            // Layer sizes
            writer.WriteStartElement("Layers");

            for (int l = 0; l < _layerCount; l++)
            {
                writer.WriteStartElement("Layer");

                writer.WriteAttributeString("Index", l.ToString());
                writer.WriteAttributeString("Size", _layerSize[l].ToString());
                writer.WriteAttributeString("Type", _transferFunction[l].ToString());

                writer.WriteEndElement();
            }

            writer.WriteEndElement(); // Layers

            writer.WriteEndElement(); // Parameters

            // Weights and Biases
            writer.WriteStartElement("Weights");

            for (int l = 0; l < _layerCount; l++)
            {
                writer.WriteStartElement("Layer");
                writer.WriteAttributeString("Index", l.ToString());

                for (int j = 0; j < _layerSize[l]; j++)
                {
                    writer.WriteStartElement("Node");
                    writer.WriteAttributeString("Index", j.ToString());
                    writer.WriteAttributeString("Bias", _bias[l][j].ToString());

                    for (int i = 0; i < (l == 0 ? _inputSize : _layerSize[l - 1]); i++)
                    {
                        writer.WriteStartElement("Axon");
                        writer.WriteAttributeString("Index", i.ToString());
                        writer.WriteString(_weight[l][i][j].ToString());

                        writer.WriteEndElement(); // Axon
                    }

                    writer.WriteEndElement(); // Node
                }

                writer.WriteEndElement(); // Layer
            }

            writer.WriteEndElement(); // Weights

            writer.WriteEndElement(); // NeuralNetwork

            writer.Flush();
            writer.Close();
        }

        public void Load(string filePath)
        {
            if (filePath == null)
                throw new ArgumentException("filePath");
            if (!File.Exists(filePath))
                throw new FileNotFoundException("filePath Not Found", filePath);

            _networkDefinition = new XmlDocument();
            _networkDefinition.Load(filePath);

            string basePath = String.Empty;
            string nodePath = String.Empty;

            // Load from xml
            if (XPathValue("NeuralNetwork/@Type") != "BackPropagation")
                throw new ApplicationException("NeuralNetwork definition not found.");

            basePath = "NeuralNetwork/Parameters/";
            NetworkName = XPathValue(basePath + "NetworkName");
            _inputSize = Convert.ToInt32(XPathValue(basePath + "InputSize"));
            _layerCount = Convert.ToInt32(XPathValue(basePath + "LayerCount"));
            _layerSize = new int[_layerCount];
            _transferFunction = new TransferFunction[_layerCount];

            basePath = basePath + "Layers/Layer";
            for (int l = 0; l < _layerCount; l++)
            {
                _layerSize[l] = Convert.ToInt32(XPathValue(basePath + "[@Index='" + l.ToString() + "']/@Size"));
                _transferFunction[l] = (TransferFunction)Enum.Parse(typeof (TransferFunction), XPathValue(basePath + "[@Index='" + l.ToString() + "']/@Type"));
            }

            // Parse weights
            _layerInput = new double[_layerCount][];
            _layerOutput = new double[_layerCount][];

            _delta = new double[_layerCount][];
            _bias = new double[_layerCount][];
            _previousBiasDelta = new double[_layerCount][];

            _weight = new double[_layerCount][][];
            _previousWeightDelta = new double[_layerCount][][];


            for (int l = 0; l < _layerCount; l++)
            {
                //Fill 2 dimensional array
                _layerInput[l] = new double[_layerSize[l]];
                _layerOutput[l] = new double[_layerSize[l]];

                _delta[l] = new double[_layerSize[l]];
                _bias[l] = new double[_layerSize[l]];
                _previousBiasDelta[l] = new double[_layerSize[l]];

                //Fill 3 dimensional array
                _weight[l] = new double[l == 0 ? _inputSize : _layerSize[l - 1]][];
                _previousWeightDelta[l] = new double[l == 0 ? _inputSize : _layerSize[l - 1]][];

                for (int i = 0; i < (l == 0 ? _inputSize : _layerSize[l - 1]); i++)
                {
                    _weight[l][i] = new double[_layerSize[l]];
                    _previousWeightDelta[l][i] = new double[_layerSize[l]];
                }
            }

            //Initialize weights

            for (int l = 0; l < _layerCount; l++)
            {
                basePath = "NeuralNetwork/Weights/Layer[@Index='" + l.ToString() + "']/";
                for (int j = 0; j < _layerSize[l]; j++)
                {
                    nodePath = "Node[@Index='" + j.ToString() + "']/@Bias";
                    _bias[l][j] = Convert.ToDouble(XPathValue(basePath + nodePath));
                    _layerOutput[l][j] = 0.0;
                    _layerInput[l][j] = 0.0;
                    _previousBiasDelta[l][j] = 0.0;
                    _delta[l][j] = 0.0;

                }
                for (int i = 0; i < (l == 0 ? _inputSize : _layerSize[l - 1]); i++)
                {
                    for (int j = 0; j < _layerSize[l]; j++)
                    {
                        nodePath = "Node[@Index='" + j.ToString() + "']/Axon[@Index='" + i.ToString() + "']";
                        _weight[l][i][j] = Convert.ToDouble(XPathValue(basePath + nodePath));
                        _previousWeightDelta[l][i][j] = 0.0;
                    }
                }
            }

            // release
            _networkDefinition = null;
        }

        public void LoadNetwork(NetworkSettingsXmlFile settings)
        {
            if (settings == null)
                throw new ArgumentException("settings");

            NetworkName = settings.NetworkName;
            _inputSize = settings.InputSize;
            _layerCount = settings.LayerCount;
            _layerSize = settings.LayerSizes;
            _transferFunction = settings.TransferFunctions;

            _layerInput = settings.LayerInput;
            _layerOutput = settings.LayerOutput;

            _delta = settings.Delta;
            _bias = settings.Bias;
            _previousBiasDelta = settings.PreviousBiasDelta;

            _weight = settings.Weight;
            _previousWeightDelta = settings.PreviousWeightDelta;
        }

        public void LoadSettings(NetworkSettingsXmlFile settings)
        {
            if (settings == null)
                throw new ArgumentException("settings");
            
            NetworkName = settings.NetworkName;
            _initLayerSizes = settings.LayerSizes;
            _initTransferFunctions = settings.TransferFunctions;

            _layerInput = (double[][])settings.LayerInput.Clone();
            _layerOutput = (double[][])settings.LayerOutput.Clone();

            Initialize();
            RandomizeWeights();
        }
        #endregion

        #region Private Methods
        private string XPathValue(string xPath)
        {
            XmlNode node = _networkDefinition.SelectSingleNode(xPath);

            if (node == null)
                throw new ArgumentException("Invalid XPath", "xPath");

            return node.InnerText;
        }

        private void RandomizeWeights()
        {
            for (int l = 0; l < _layerCount; l++)
            {
                for (int j = 0; j < _layerSize[l]; j++)
                {
                    _bias[l][j] = Gaussian.GetRandomGaussian();
                    _layerOutput[l][j] = 0.0;
                    _layerInput[l][j] = 0.0;
                    _previousBiasDelta[l][j] = 0.0;
                    _delta[l][j] = 0.0;

                }
                for (int i = 0; i < (l == 0 ? _inputSize : _layerSize[l - 1]); i++)
                {
                    for (int j = 0; j < _layerSize[l]; j++)
                    {
                        _weight[l][i][j] = Gaussian.GetRandomGaussian();
                        _previousWeightDelta[l][i][j] = 0.0;
                    }
                }
            }
        }
        #endregion
    }
}
