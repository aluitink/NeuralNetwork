using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using Helvetica.NeuralNetwork.Models;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Helvetica.NeuralNetwork.Test
{
    [TestClass]
    public class BackPropagationNetworkTest
    {
        [TestMethod]
        public void CanTrainXORNetwork()
        {
            int[] layerSizes = new int[4] { 2, 5, 5, 1 };
            TransferFunction[] transferFunctions = new TransferFunction[4]
                                                       {
                                                           TransferFunction.None,
                                                           TransferFunction.RationalSigmoid,
                                                           TransferFunction.Gaussian,
                                                           TransferFunction.Linear
                                                       };

            BackPropagationNetwork bpn = new BackPropagationNetwork(layerSizes, transferFunctions);

            double[][] input, output;

            input = new double[4][];
            output = new double[4][];
            for (int i = 0; i < 4; i++)
            {
                input[i] = new double[2];
                output[i] = new double[1];
            }

            input[0][0] = 0.0; input[0][1] = 0.0; output[0][0] = 0.0; // false xor false = false
            input[1][0] = 1.0; input[1][1] = 0.0; output[1][0] = 1.0; // true xor false = true
            input[2][0] = 0.0; input[2][1] = 1.0; output[2][0] = 1.0; // false xor true = true
            input[3][0] = 1.0; input[3][1] = 1.0; output[3][0] = 0.0; // true xor true = false

            double error = 0.0;
            int count = 0, maxCount = 1000000;

            do
            {
                count++;
                error = 0.0;

                for (int i = 0; i < 4; i++)
                    error += bpn.Train(ref input[i], ref output[i], 0.25, 0.3);

                if (count % 100 == 0)
                    Console.WriteLine("Epoch {0} completed with error {1:0.0001}", count, error);

            } while (error > 0.0001 && count <= maxCount);


        }

        [TestMethod]
        public void CanTrainNetwork()
        {
            int[] layerSizes = new int[3] { 1, 2, 1 };
            TransferFunction[] transferFunctions = new TransferFunction[3]
                                                       {
                                                           TransferFunction.None,
                                                           TransferFunction.RationalSigmoid,
                                                           TransferFunction.Linear
                                                       };

            BackPropagationNetwork bpn = new BackPropagationNetwork(layerSizes, transferFunctions);

            //desired output needs to be between 0 - 1
            double[] input = new double[1] { 1.0 };
            double[] desired = new double[1] { 2.5 };

            double[] output = new double[1];

            double error = 0.0;


            for (int i = 0; i < 1000; i++)
            {
                error = bpn.Train(ref input, ref desired, 0.15, 0.1);
                bpn.Run(ref input, out output);

                if (i % 100 == 0)
                    Console.WriteLine("Itteration {0}:\r\nInput {1:0.000} Output {2:0.000} Error {3:0.000}", i, input[0], output[0], error);
            }

        }

        [TestMethod]
        public void CanSaveNetwork()
        {
            int[] layerSizes = new int[3] { 1, 2, 1 };
            TransferFunction[] transferFunctions = new TransferFunction[3]
                                                       {
                                                           TransferFunction.None,
                                                           TransferFunction.RationalSigmoid,
                                                           TransferFunction.Linear
                                                       };

            BackPropagationNetwork bpn = new BackPropagationNetwork(layerSizes, transferFunctions);

            String path = Path.GetTempFileName();
            bpn.Save(path);

            Assert.IsTrue(File.Exists(path));
        }



        [TestMethod]
        public void CanLoadNetwork()
        {
            int[] layerSizes = new int[3] { 1, 2, 1 };
            TransferFunction[] transferFunctions = new TransferFunction[3]
                                                       {
                                                           TransferFunction.None,
                                                           TransferFunction.RationalSigmoid,
                                                           TransferFunction.Linear
                                                       };

            BackPropagationNetwork bpn = new BackPropagationNetwork(layerSizes, transferFunctions);

            //desired output needs to be between 0 - 1
            double[] input = new double[1] { 1.0 };
            double[] desired = new double[1] { 2.5 };

            double[] output = new double[1];

            double error = 0.0;


            for (int i = 0; i < 1000; i++)
            {
                error = bpn.Train(ref input, ref desired, 0.15, 0.1);
                bpn.Run(ref input, out output);

                if (i % 100 == 0)
                    Console.WriteLine("Itteration {0}:\r\nInput {1:0.000} Output {2:0.000000} Error {3:0.000}", i, input[0], output[0], error);
            }

            String path = Path.GetTempFileName();
            bpn.Save(path);

            Assert.IsTrue(File.Exists(path));

            BackPropagationNetwork bpn2 = new BackPropagationNetwork(path);
            double[] outputActual = new double[1];
            bpn2.Run(ref input, out outputActual);

            Console.WriteLine("Input {0:0.000} Output {1:0.000000} Error {2:0.000}", input[0], outputActual[0], error);
            Assert.AreEqual(output[0].ToString("0.000"), outputActual[0].ToString("0.000"));
        }
    }
}
