using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using MulticlassClassifier;
using Xunit;
using Xunit.Abstractions;

namespace MulticlassClassificationSample
{
    public class MainTester
    {
        private readonly ITestOutputHelper _output;
        private readonly string _dataFolderPath;

        public MainTester(ITestOutputHelper output)
        {
            this._output = output;
            this._dataFolderPath = Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location), "..\\..\\..\\..\\data\\");
        }
        
        [Fact]
        public void Test_Check_Data_Folder()
        {
            _output.WriteLine($"Data Folder Path : {this._dataFolderPath}");
        }
        
        [Fact]
        public void Test_Evaluate_Prediction_Model_Score()
        {
            _output.WriteLine("Test_Evaluate_Prediction_Model_Score()...");

            var predictor = new Predictor();
            
            predictor.LoadTrainData(Path.Combine(_dataFolderPath,"issues_train.tsv"));
            predictor.LoadTestData(Path.Combine(_dataFolderPath,"issues_test.tsv"));
            
            predictor.BuildAndTrainModel();
            
            var metrics = predictor.EvaluateModelMetrics();
            
            _output.WriteLine($"Model Metrics - MicroAccuracy:{metrics.MicroAccuracy:P2}, MacroAccuracy:{metrics.MacroAccuracy:P2}, LogLoss:{metrics.LogLoss:P2}, LogLossReduction:{metrics.LogLossReduction:P2}");
            
            Assert.InRange(metrics.MicroAccuracy,0.5, 0.95);
            Assert.InRange(metrics.MacroAccuracy,0.5, 0.95);

            _output.WriteLine("...Test_Evaluate_Prediction_Model_Score() DONE.");
        }
        
        [Fact]
        public void Test_Predict_Area_System_Data()
        {
            _output.WriteLine("Test_Predict_Area_System_Data()...");

            const string INPUT_DATA_TITLE = "DataTable not updating Database with DataAdapter";
            const string INPUT_DATA_DESCRIPTION = @"I am trying to update a database with info from a WinForm. I had no issues when using a “normal” SQL update command written by hand " +
                                                  @"(parameters set to the text box values,) but I am trying to clean up and reduce my code and I thought I would bind the controls to a DataTable" +
                                                  @" and use a DataAdapter's update command to achieve the same thing. I have tried to get various combinations of setting parameters and update" +
                                                  @" commands to work, but the Database is not getting updated from the new DataTable values. I have stepped through the code with each change and can" +
                                                  @" see that the DataTable is getting the new textbox values, but those updates aren’t going to the Database.";
            
            var predictor = new Predictor();
            
            predictor.LoadTrainData(Path.Combine(_dataFolderPath,"issues_train.tsv"));

            predictor.BuildAndTrainModel();

            var prediction = predictor.Predict(INPUT_DATA_TITLE, INPUT_DATA_DESCRIPTION);
            
            _output.WriteLine($"Prediction for \"{INPUT_DATA_TITLE}\" : {prediction}");
            
            Assert.Equal("area-System.Data",prediction);
            
            _output.WriteLine("...Test_Predict_Area_System_Data() DONE.");
        }
        
        [Fact]
        public void Test_Predict_Area_System_Data_With_Saved_And_Reloaded_Model()
        {
            _output.WriteLine("Test_Predict_Area_System_Data_With_Saved_And_Reloaded_Model()...");

            const string INPUT_DATA_TITLE = "DataTable not updating Database with DataAdapter";
            const string INPUT_DATA_DESCRIPTION = @"I am trying to update a database with info from a WinForm. I had no issues when using a “normal” SQL update command written by hand " +
                                                  @"(parameters set to the text box values,) but I am trying to clean up and reduce my code and I thought I would bind the controls to a DataTable" +
                                                  @" and use a DataAdapter's update command to achieve the same thing. I have tried to get various combinations of setting parameters and update" +
                                                  @" commands to work, but the Database is not getting updated from the new DataTable values. I have stepped through the code with each change and can" +
                                                  @" see that the DataTable is getting the new textbox values, but those updates aren’t going to the Database.";
            
            var predictor = new Predictor();
            
            var sw = new Stopwatch();
            
            predictor.LoadTrainData(Path.Combine(_dataFolderPath,"issues_train.tsv"));
            
            sw.Start();

            predictor.BuildAndTrainModel();
            
            _output.WriteLine($"Tempo per generare il modello: {sw.ElapsedMilliseconds} ms");

            var modelFilePath = Path.Combine(_dataFolderPath, "TheModel.model");
            
            if(File.Exists(modelFilePath)) File.Delete(modelFilePath);
            
            predictor.SaveModel(modelFilePath);
            
            sw.Reset();

            predictor.LoadModel(modelFilePath);

            _output.WriteLine($"Tempo per caricare il modello: {sw.ElapsedMilliseconds} ms");
            
            sw.Reset();

            var prediction = predictor.Predict(INPUT_DATA_TITLE, INPUT_DATA_DESCRIPTION);
            
            _output.WriteLine($"Tempo per effettuare la predizione: {sw.ElapsedMilliseconds} ms");
            
            _output.WriteLine($"Prediction for \"{INPUT_DATA_TITLE}\" : {prediction}");
            
            Assert.Equal("area-System.Data",prediction);
            
            _output.WriteLine("...Test_Predict_Area_System_Data_With_Saved_And_Reloaded_Model() DONE.");
        }
    }
}