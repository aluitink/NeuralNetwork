using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Security.Authentication.ExtendedProtection;
using System.Text;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Helvetica.NeuralNetwork.Test
{
    //public interface ISequenceFilter
    //{
    //    double[,] Apply(double[,] data);
    //}

    //public class SequenceNetwork
    //{
    //    protected List<ISequenceFilter> Filters = new List<ISequenceFilter>(); 

    //    public void Add(ISequenceFilter filter)
    //    {
    //        Filters.Add(filter);
    //    }

    //    public double[,] ApplySequence(Bitmap rawImage)
    //    {
    //        double[,] imageData = GetDataMap(rawImage);

    //        foreach (ISequenceFilter sequenceFilter in Filters)
    //        {
    //            imageData = sequenceFilter.Apply(imageData);
    //        }

    //        return imageData;
    //    }

    //    public double[,] GetDataMap(Bitmap bitmap)
    //    {
    //        return BitmapUtilities.ImageToMap(bitmap, BitmapUtilities.RGBToYChannel_VD);
    //    }

    //    public Bitmap GetImage(double[,] dataMap)
    //    {
    //        return BitmapUtilities.MapToImage(dataMap);
    //    }

    //    public void Reset()
    //    {
    //        Filters.Clear();
    //    }
    //}

    //public class GausianSequenceFilter: ISequenceFilter
    //{

    //    public double[,] Calculate(double[,] input, int length, double weight)
    //    {
    //        double sumTotal = 0;
    //        int kernelRadius = length / 2;
    //        double distance = 0;


    //        double calculatedEuler = 1.0 /
    //        (2.0 * Math.PI * Math.Pow(weight, 2));


    //        for (int filterY = -kernelRadius;
    //             filterY <= kernelRadius; filterY++)
    //        {
    //            for (int filterX = -kernelRadius;
    //                filterX <= kernelRadius; filterX++)
    //            {
    //                distance = ((filterX * filterX) +
    //                           (filterY * filterY)) /
    //                           (2 * (weight * weight));


    //                input[filterY + kernelRadius,
    //                       filterX + kernelRadius] =
    //                       calculatedEuler * Math.Exp(-distance);


    //                sumTotal += input[filterY + kernelRadius,
    //                                   filterX + kernelRadius];
    //            }
    //        }


    //        for (int y = 0; y < length; y++)
    //        {
    //            for (int x = 0; x < length; x++)
    //            {
    //                double oldValue = input[y, x];
    //                double newValue = input[y, x] * (1.0 / sumTotal);

    //                Console.WriteLine("OldValue: {0}, NewValue: {1}", oldValue, newValue);

    //                input[y, x] = newValue;

    //            }
    //        }


    //        return input;
    //    }

    //    public double[,] Apply(double[,] data)
    //    {
    //        return Calculate(data, 5, 7);
    //    }
    //}


    //[TestClass]
    //public class ConvolutionTest
    //{
    //    private const string ResourcePath = "Helvetica.NeuralNetwork.Test.Resources.";
    //    [TestMethod]
    //    public void ImageTest()
    //    {
    //        Bitmap image = new Bitmap(GetResource("face1_source.jpg"));


    //        SequenceNetwork sn = new SequenceNetwork();
    //        sn.Add(new GausianSequenceFilter());

    //        double[,] imageMap = sn.ApplySequence(image);

    //        Bitmap image2 = sn.GetImage(imageMap);
    //        image2.Save("image2.jpg");
    //    }
        

    //    public void ImageInput()
    //    {
    //        //Get Input
    //        //Find Face
    //        //Get YUV Image
    //        //Get Y Channel
    //        //Bitmap input = new Bitmap("");
    //        //Rectangle faceRec = DetectFace(input);
    //        //Bitmap faceImage = input.Clone(faceRec, input.PixelFormat);

    //        //Bitmap image = GetYUVImage(input);

    //        //Bitmap yChannel = GetImageChannel(image, Channel.Y);

    //        //Bitmap filtered = FilterGausian(yChannel, 7);

    //        //Bitmap convoluted = SpatialConvolution(filtered, 1, 8, 5, 5);


    //    }


    //    private Stream GetResource(string file)
    //    {
    //        return this.GetType().Assembly.GetManifestResourceStream(ResourcePath + file);
    //    }
    //}


}
