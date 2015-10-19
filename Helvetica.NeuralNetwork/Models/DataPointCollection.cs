using System;
using System.Collections.Generic;
using System.Xml.Serialization;

namespace Helvetica.NeuralNetwork.Models
{
    [Serializable]
    [XmlRoot("DataPointCollection")]
    public class DataPointCollection: List<DataPoint>
    {
        public int DataPointBound { get { return this[0] != null ? this[0].Input.Length : 0; } }
    }
}
