using System;

namespace Helvetica.NeuralNetwork
{
    public class Permutator
    {
        private Random _random = new Random();
        private int[] _index;

        public Permutator(int size)
        {
            _index = new int[size];
            for (int i = 0; i < size; i++)
                _index[i] = i;
            Permute(size);
        }
        public void Permute(int nTimes)
        {
            int i, j, t;
            for (int n = 0; n < nTimes; n++)
            {
                i = _random.Next(_index.Length);
                j = _random.Next(_index.Length);

                if(i!=j)
                {
                    t = _index[i];
                    _index[i] = _index[j];
                    _index[j] = t;
                }
            }

        }
        public int this[int i] { get { return _index[i]; } }
    }
}
