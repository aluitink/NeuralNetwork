using System;
using System.Xml.Serialization;
using Helvetica.NeuralNetwork.Models;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Helvetica.NeuralNetwork.Test
{
    [TestClass]
    public class DataPointSerializationTest
    {
        [TestMethod]
        public void DataPoint_CanSerailize()
        {
            DataPoint point1 = new DataPoint(new[] { 1.0, 1.1 }, new[] { 2.0, 2.2 });
            DataPoint point2 = new DataPoint(new[] { 2.0, 2.1 }, new[] { 4.0, 2.2 });
            DataPoint point3 = new DataPoint(new[] { 3.0, 3.1 }, new[] { 6.0, 2.2 });
            DataPoint point4 = new DataPoint(new[] { 4.0, 4.1 }, new[] { 8.0, 2.2 });
            DataPoint point5 = new DataPoint(new[] { 5.0, 5.1 }, new[] { 10.0, 2.2 });

            DataPointCollection dataPointCollection = new DataPointCollection();

            dataPointCollection.Add(point1);
            dataPointCollection.Add(point2);
            dataPointCollection.Add(point3);
            dataPointCollection.Add(point4);
            dataPointCollection.Add(point5);
            XmlSerializer serializer = new XmlSerializer(typeof(DataPointCollection));
            serializer.Serialize(Console.Out, dataPointCollection);

        }
    }
}
