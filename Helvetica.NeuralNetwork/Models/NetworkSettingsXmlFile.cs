using System;
using System.IO;
using System.Xml;

namespace Helvetica.NeuralNetwork.Models
{
    public class NetworkSettingsXmlFile: NetworkSettingsBase
    {
        private XmlDocument _networkDefinition;

        public NetworkSettingsXmlFile()
        {

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
            writer.WriteElementString("InputSize", InputSize.ToString());
            writer.WriteElementString("LayerCount", LayerCount.ToString());

            // Layer sizes
            writer.WriteStartElement("Layers");

            for (int l = 0; l < LayerCount; l++)
            {
                writer.WriteStartElement("Layer");

                writer.WriteAttributeString("Index", l.ToString());
                writer.WriteAttributeString("Size", LayerSizes[l].ToString());
                writer.WriteAttributeString("Type", TransferFunctions[l].ToString());

                writer.WriteEndElement();
            }

            writer.WriteEndElement(); // Layers

            writer.WriteEndElement(); // Parameters

            // Weights and Biases
            writer.WriteStartElement("Weights");

            for (int l = 0; l < LayerCount; l++)
            {
                writer.WriteStartElement("Layer");
                writer.WriteAttributeString("Index", l.ToString());

                for (int j = 0; j < LayerSizes[l]; j++)
                {
                    writer.WriteStartElement("Node");
                    writer.WriteAttributeString("Index", j.ToString());
                    writer.WriteAttributeString("Bias", Bias[l][j].ToString());

                    for (int i = 0; i < (l == 0 ? InputSize : LayerSizes[l - 1]); i++)
                    {
                        writer.WriteStartElement("Axon");
                        writer.WriteAttributeString("Index", i.ToString());
                        writer.WriteString(Weight[l][i][j].ToString());

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
            InputSize = Convert.ToInt32(XPathValue(basePath + "InputSize"));
            LayerCount = Convert.ToInt32(XPathValue(basePath + "LayerCount"));
            LayerSizes = new int[LayerCount];
            TransferFunctions = new TransferFunction[LayerCount];

            basePath = basePath + "Layers/Layer";
            for (int l = 0; l < LayerCount; l++)
            {
                LayerSizes[l] = Convert.ToInt32(XPathValue(basePath + "[@Index='" + l.ToString() + "']/@Size"));
                TransferFunctions[l] = (TransferFunction)Enum.Parse(typeof(TransferFunction), XPathValue(basePath + "[@Index='" + l.ToString() + "']/@Type"));
            }

            // Parse weights
            LayerInput = new double[LayerCount][];
            LayerOutput = new double[LayerCount][];

            Delta = new double[LayerCount][];
            Bias = new double[LayerCount][];
            PreviousBiasDelta = new double[LayerCount][];

            Weight = new double[LayerCount][][];
            PreviousWeightDelta = new double[LayerCount][][];


            for (int l = 0; l < LayerCount; l++)
            {
                //Fill 2 dimensional array
                LayerInput[l] = new double[LayerSizes[l]];
                LayerOutput[l] = new double[LayerSizes[l]];

                Delta[l] = new double[LayerSizes[l]];
                Bias[l] = new double[LayerSizes[l]];
                PreviousBiasDelta[l] = new double[LayerSizes[l]];

                //Fill 3 dimensional array
                Weight[l] = new double[l == 0 ? InputSize : LayerSizes[l - 1]][];
                PreviousWeightDelta[l] = new double[l == 0 ? InputSize : LayerSizes[l - 1]][];

                for (int i = 0; i < (l == 0 ? InputSize : LayerSizes[l - 1]); i++)
                {
                    Weight[l][i] = new double[LayerSizes[l]];
                    PreviousWeightDelta[l][i] = new double[LayerSizes[l]];
                }
            }

            //Initialize weights

            for (int l = 0; l < LayerCount; l++)
            {
                basePath = "NeuralNetwork/Weights/Layer[@Index='" + l.ToString() + "']/";
                for (int j = 0; j < LayerSizes[l]; j++)
                {
                    nodePath = "Node[@Index='" + j.ToString() + "']/@Bias";
                    Bias[l][j] = Convert.ToDouble(XPathValue(basePath + nodePath));
                    LayerOutput[l][j] = 0.0;
                    LayerInput[l][j] = 0.0;
                    PreviousBiasDelta[l][j] = 0.0;
                    Delta[l][j] = 0.0;

                }
                for (int i = 0; i < (l == 0 ? InputSize : LayerSizes[l - 1]); i++)
                {
                    for (int j = 0; j < LayerSizes[l]; j++)
                    {
                        nodePath = "Node[@Index='" + j.ToString() + "']/Axon[@Index='" + i.ToString() + "']";
                        Weight[l][i][j] = Convert.ToDouble(XPathValue(basePath + nodePath));
                        PreviousWeightDelta[l][i][j] = 0.0;
                    }
                }
            }

            // release
            _networkDefinition = null;
        }

        #region Private Methods
        private string XPathValue(string xPath)
        {
            XmlNode node = _networkDefinition.SelectSingleNode(xPath);

            if (node == null)
                throw new ArgumentException("Invalid XPath", "xPath");

            return node.InnerText;
        }
        #endregion
    }
}
