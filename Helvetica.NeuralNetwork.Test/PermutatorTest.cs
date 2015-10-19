using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Helvetica.NeuralNetwork.Test
{
    [TestClass]
    public class PermutatorTest
    {
        [TestMethod]
        public void Permutator_CanPermutate()
        {
            Permutator idx = new Permutator(10);

            for (int i = 0; i < 10; i++)
                Console.WriteLine("{0}", idx[i]);
        }
    }
}
