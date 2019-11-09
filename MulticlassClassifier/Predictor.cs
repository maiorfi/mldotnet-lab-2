using System.ComponentModel.DataAnnotations;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MulticlassClassifier
{
    public class Predictor
    {
        private static MLContext _mlContext = new MLContext();

        private static ITransformer _model;

        static IDataView _trainingData;
        static IDataView _testData;

        public void LoadTrainData(string dataFilePath)
        {
            _trainingData = _mlContext.Data.LoadFromTextFile<GitHubIssue>(dataFilePath, hasHeader: true);
        }

        public void LoadTestData(string dataFilePath)
        {
            _testData = _mlContext.Data.LoadFromTextFile<GitHubIssue>(dataFilePath, hasHeader: true);
        }

        public void BuildAndTrainModel()
        {
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))
                .Append(_mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
                .AppendCacheCheckpoint(_mlContext);

            var trainingPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            _model = trainingPipeline.Fit(_trainingData);
        }

        public MulticlassClassificationMetrics EvaluateModelMetrics()
        {
            var testMetrics = _mlContext.MulticlassClassification.Evaluate(_model.Transform(_testData));

            return testMetrics;
        }

        public string Predict(string title, string description)
        {
            var predictionFunction = _mlContext.Model.CreatePredictionEngine<GitHubIssue, GitHubIssuePrediction>(_model);

            var resultPrediction = predictionFunction.Predict(new GitHubIssue()
            {
                Title = title,
                Description = description
            });

            return resultPrediction.Area;
        }

        public void SaveModel(string modelFilePath)
        {
            _mlContext.Model.Save(_model, _trainingData.Schema, modelFilePath);
        }
        
        public void LoadModel(string modelFilePath)
        {
            _model = _mlContext.Model.Load(modelFilePath, out var dataViewSchema);
        }
    }
}