using System;
using System.Collections.Generic;
using Helvetica.NeuralNetwork.Models;

namespace Helvetica.NeuralNetwork
{
    public delegate void NetworkTrainerFinishedEventHandler(object sender, EventArgs e);
    public delegate void NetworkTrainerTickEventHandler(object sender, NetworkTrainerEventArgs e);

    public enum NetworkTrainerStatus
    {
        Idle,
        Running,
        Stopped,
        Error,
        Finished
    }

    public class NetworkTrainerEventArgs: EventArgs
    {
        public NetworkTrainerStatus Status { get; set; }
    }

    public class NetworkTrainer
    {
        private double _error;
        private Permutator _idx;

        public BackPropagationNetwork Network { get; private set; }
        public DataPointCollection DataSet { get; private set; }
        public List<double> ErrorHistory { get; private set; }
        public double TargetError { get; set; }
        public double SmallestError { get; set; }
        public double MaxIterations { get; set; }
        public double TrainingRate { get; set; }
        public double Momentum { get; set; }
        public double NudgeScale { get; set; }
        public double NudgeTolerance { get; set; }
        public int NudgeWindow { get; set; }
        public int CurrentIteration { get; private set; }
        public bool Nudge { get; set; }

        public NetworkTrainerFinishedEventHandler Finished;
        public NetworkTrainerTickEventHandler Tick;

        public NetworkTrainer(BackPropagationNetwork network, DataPointCollection dataSet)
        {
            Nudge = true;
            NudgeTolerance = 0.0001;
            NudgeScale = 0.4;
            NudgeWindow = 100;
            
            TargetError = 0.001;
            MaxIterations = 100000;
            
            TrainingRate = 0.15;
            Momentum = 0.10;
            
            Network = network;
            DataSet = dataSet;
            
            Initialize();
        }

        public void Initialize()
        {
            if(_idx == null)
                _idx = new Permutator(DataSet.Count);
            else
                _idx.Permute(DataSet.Count);

            if(ErrorHistory == null)
                ErrorHistory = new List<double>();
            else
                ErrorHistory.Clear();
        }

        public void TrainNetwork()
        {
            OnTick(new NetworkTrainerEventArgs()
            {
                Status = NetworkTrainerStatus.Running
            });

            do
            {
                BeforeTrainDataSet();
                TrainDataSet();
                AfterTrainDataSet();
            } while (_error > TargetError && CurrentIteration < MaxIterations);

            OnTick(new NetworkTrainerEventArgs()
            {
                Status = NetworkTrainerStatus.Finished
            });
        }

        protected virtual void BeforeTrainEpoch() { }
        protected virtual void AfterTrainEpoch()
        {
            OnTick(new NetworkTrainerEventArgs()
            {
                Status = NetworkTrainerStatus.Running
            });
        }

        protected virtual void BeforeTrainDataPoint(ref double[] input, ref double[] output, int index) { }
        protected virtual void AfterTrainDataPoint(ref double[] input, ref double[] output, int index) { }

        protected void OnFinished(EventArgs eventArgs)
        {
            if (Finished != null)
                Finished(this, eventArgs);
        }

        protected void OnTick(NetworkTrainerEventArgs eventArgs)
        {
            if(Tick != null)
                Tick(this, eventArgs);
        }

        private void BeforeTrainDataSet()
        {
            CurrentIteration++;
            _error = 0.0;
            _idx.Permute(DataSet.Count);
            BeforeTrainEpoch();
        }

        private void TrainDataSet()
        {
            for (int i = 0; i < DataSet.Count; i++)
            {
                double[] input = (double[])DataSet[_idx[i]].Input.Clone();
                double[] output = (double[])DataSet[_idx[i]].Output.Clone();

                BeforeTrainDataPoint(ref input, ref output, _idx[i]);
                _error += Network.Train(ref input, ref output, TrainingRate, Momentum);
                AfterTrainDataPoint(ref input, ref output, _idx[i]);
            }
        }

        private void AfterTrainDataSet()
        {
            ErrorHistory.Add(_error);

            // Check if Nudge is needed
            if (CurrentIteration % NudgeWindow == 0 && Nudge)
                CheckNudge();
            AfterTrainEpoch();
        }

        private void CheckNudge()
        {            
            // Enough Data?
            if (CurrentIteration < 2 * NudgeWindow) return;

            double oldAverage = 0.0;
            double newAverage = 0.0;
            int historyLength = ErrorHistory.Count;

            // Compute Averages
            for (int i = 0; i < NudgeWindow; i++)
            {
                oldAverage += ErrorHistory[historyLength - 2 * NudgeWindow + i];
                newAverage += ErrorHistory[historyLength - NudgeWindow + i];
            }

            oldAverage /= NudgeWindow;
            newAverage /= NudgeWindow;

            if (((double)Math.Abs(newAverage - oldAverage) / NudgeWindow < NudgeTolerance))
            {
                Network.Nudge(NudgeScale);
                Console.WriteLine(" Nudge");
            }
            
        }
    }
}
