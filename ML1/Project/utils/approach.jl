
include("libraries.jl")
include("preprocess.jl")
include("dimensionality.jl")
include("ensembles.jl")
include("models.jl")
include("postProcess.jl")


# Timer object to get run times per function.

to = TimerOutput()
#disable_timer!(to)


# Main functions.

"""Creates an approach given a configuration.""" 
function createApproach(;name::String="", multiplyClassSamples=[], anns=[], svms=[], dts=[], knns=[], useKFoldCrossValidation::Bool=false, 
    kFolds::Int=5, reduceDimensionality::Bool=false, dimensionalityReduction::DimensionalityReduction=DR_PCA, ensembles=[],
    numberBestModels=5, validationRatio::Float64=0.0, reportFolder::String="./result/", showPlots::Bool=false, dimReduction::Float64=2.0,
    customs=[], colsSelected::UnitRange{Int}=1:1
)::Approach
    return Approach(name, multiplyClassSamples, anns, svms, dts, knns, useKFoldCrossValidation, kFolds,
            reduceDimensionality, dimensionalityReduction, ensembles, numberBestModels, validationRatio, 
            reportFolder, showPlots, dimReduction, customs, colsSelected)
end

"""Runs a given approach over a given problem, training and evaluating all the models.""" 
function run(problem::Problem, approach::Approach)
    @timeit to "preprocessDataset" begin
        dataset = preprocessDataset(problem, approach)
    end
    show(to)
    println()
    @timeit to "normalizeDataset" begin        
        normalized = normalizeDataset(approach, dataset)
    end
    show(to)
    println()
    @timeit to "trainAndValidateModels" begin     
    models = trainAndValidateModels(approach, normalized)
    end
    show(to)
    println()   
    @timeit to "evaluateModels" begin            
    results = evaluateModels(problem, approach, normalized, models)
    end
    show(to)
    println()
    for m=models
        println("########################################################")
        println(m.name)
        println(m.validationMetrics.classificationReport)
    end
    printResults(results)
    storeResultData(approach.name, results; folder=approach.reportFolder)
    return results;
end


# Internal functions.

"""Reads and pre-processes the datasets according to the problem and approach settings.
- Converts a multi-class problem in a binary one if necessary.
- Reduces de dimensionality of the dataset if necessary.
- Reduces the number of samples per class if necessary."""
function preprocessDataset(problem::Problem, approach::Approach)::Dataset
    train, test = readDatasets(problem.trainSetPath, problem.testSetPath)
    classColumn = size(train)[2]
    if problem.isBinaryClassification
        train, test = convertInBinaryProblem!(problem, classColumn, train, test)
    end
    if (approach.reduceDimensionality && approach.dimensionalityReduction == CUSTOM)
        train, test = reducesDimensionalityCustom(approach.colsSelected, train, test)
    end       
    if approach.reduceDimensionality
        train, test = reduceDimensionality(approach.dimensionalityReduction, train, test, dimReduction=approach.dimReduction)
        classColumn = size(train)[2]
    end
    trainProcessed, original, final = preprocessTrainDataset(
        train,
        class_column=classColumn,
        multiply_class_samples=approach.multiplyClassSamples
    )
    dataset = Dataset(train, trainProcessed, original, final, test)
    return dataset
end

"""Converts the train and test sets for the given multi-class problem in binary classification sets
by transforming the column with the class value."""
function convertInBinaryProblem!(problem::Problem, classColumn::Int, trainSet::DataFrame, testSet::DataFrame)
    if !problem.isBinaryClassification
        return trainSet, testSet
    end
    positiveClassValues = map(x -> Int(x), problem.positiveClasses)
    trainSet[:,classColumn] = [(x in positiveClassValues) for x in trainSet[:,classColumn]]
    testSet[:,classColumn] = [(x in positiveClassValues) for x in testSet[:,classColumn]]
    return trainSet, testSet
end

"""Applies the configured normalization to the dataset and returns a prepared dataset."""
function normalizeDataset(approach::Approach, dataset::Dataset)::PreparedDataset
    trainData = Matrix(dataset.trainProcessed)
    trainInputs = normalizeMinMax!(trainData[:,1:end-1])
    trainTargets = vec(trainData[:,end])
    testData = Matrix(dataset.test)
    testInputs = normalizeMinMax!(testData[:,1:end-1])
    testTargets = vec(testData[:,end])
    return PreparedDataset(trainInputs, trainTargets, testInputs, testTargets)
end

"""Trains and validates all the models specified by the approach on the provided prepared dataset.
First, the models are created.
Then all the simple models are trained and validated.
Finally all the ensemble models are trained and validated."""
function trainAndValidateModels(approach::Approach, dataset::PreparedDataset)
    models, ensembles = createModels(approach)
    trainedModels = trainModels(approach, dataset, models)
    trainedEnsembles = trainEnsembles(approach, dataset, ensembles, trainedModels)
    allModels = trainedModels
    append!(allModels, trainedEnsembles)
    return allModels
end

"""Evaluates all the trained models by calculating the configured metrics on the test set."""
function evaluateModels(problem::Problem, approach::Approach, dataset::PreparedDataset, models::Array{TrainedModel})::Vector{Evaluation}
    scores::Vector{Evaluation} = []
    for trainedModel=models
        @timeit to trainedModel.name begin
        modelScore = score(trainedModel.model, dataset.testInputs, vec(dataset.testTargets))
        testOutputs = predict(trainedModel.model, dataset.testInputs) 
        testTargets = dataset.testTargets
        metrics = calculateMetrics(testOutputs, testTargets)
        if problem.isBinaryClassification
            testOutputs = testOutputs .== 1
            testTargets = testTargets .== 1
            confMat = confusionMatrix(testOutputs, testTargets) 
        else
            confMat = confusionMatrix(testOutputs, testTargets, weighted=false) 
        end
        push!(scores, Evaluation(trainedModel, modelScore, confMat, metrics))
        end
        show(to)
        println()                
    end
    return scores
end

"""Creates, trains and validates all the ensemble models.
The best models according to the validation metrics are used for the ensembles that require base models.
"""
function trainEnsembles(approach::Approach, dataset::PreparedDataset, ensembles::Array{Ensemble}, trainedModels::Array{TrainedModel})
    sortedTrainedModels = trainedModels
    sort!(sortedTrainedModels, by=m->m.validationScore, rev=true)
    bestModels = sortedTrainedModels[1:min(approach.numberBestModels,end)]
    ensembleModels = createEnsembles(ensembles, bestModels)
    trainedEnsembles = trainModels(approach, dataset, ensembleModels)
    return trainedEnsembles
end

"""Trains and validates all the provided models on the given training set.
If the approach specifies k-fold cross-validation, it is applied.
All the trained models are returned."""
function trainModels(approach::Approach, dataset::PreparedDataset, models::Array{Model})::Array{TrainedModel}
    if approach.useKFoldCrossValidation
        return trainModelsCrossValidation(approach, dataset, models)
    end
    trainInputs, validationInputs, trainTargets, validationTargets = train_test_split(
        dataset.trainInputs, dataset.trainTargets, test_size=approach.validationRatio
    )
    trainedModels::Array{TrainedModel} = []
    for model=models
        @timeit to model.name begin
            fit!(model.model, trainInputs, vec(trainTargets))
            trainingAccuracy = score(model.model, trainInputs, trainTargets)
            trainingMetrics = calculateMetrics(model.model, trainInputs, vec(trainTargets))
            if approach.validationRatio > 0
                validationAccuracy = score(model.model, validationInputs, validationTargets)
                validationMetrics = calculateMetrics(model.model, validationInputs, vec(validationTargets))
            end
            trainedModel = TrainedModel(model.model, model.name, trainingAccuracy, 
                validationAccuracy, trainingMetrics, validationMetrics)
            push!(trainedModels, trainedModel)
        end
        show(to)
        println()
    end
    return trainedModels
end

"""Calculates the configured metrics for the given trained model and 
the data set split provided as inputs and targets."""
function calculateMetrics(model, inputs, targets)
    outputs = predict(model, inputs)
    return calculateMetrics(outputs, targets)
end

"""Calculates the configured metrics for the given data set split provided as inputs and targets."""
function calculateMetrics(outputs, targets)
    accuracy = accuracy_score(targets, outputs)
    errorRate = 1-accuracy
    f1_scores = f1_score(targets, outputs, average="macro")
    precision_scores = precision_score(targets, outputs, average="macro")
    recall_scores = recall_score(targets, outputs, average="macro")
    return buildMetrics(
        accuracy=accuracy,
        errorRate=errorRate,
        fScore=mean(f1_scores),
        fScoreSTD=std(f1_scores),
        positivePredictiveValue=mean(precision_scores),
        positivePredictiveValueSTD=std(precision_scores),
        sensitivity=mean(recall_scores),
        sensitivitySTD=std(recall_scores),
        classificationReport=classification_report(targets, outputs)
    )
end

"""Trains and validates the provided models applying k-fold cross-validation according
to the specification for the approach."""
function trainModelsCrossValidation(approach::Approach, dataset::PreparedDataset, models::Array{Model})::Array{TrainedModel}
    trainedModels::Array{TrainedModel} = []
    for model=models
        @timeit to model.name begin
            scores = cross_validate(
                model.model, dataset.trainInputs, dataset.trainTargets, cv=approach.kFolds,
                scoring=("accuracy", "precision_macro", "recall_macro", "f1_macro"), return_train_score=true
            )
            fit!(model.model, dataset.trainInputs, vec(dataset.trainTargets))            
            trainingMetrics, validationMetrics = getCrossValidationMetrics(model, scores)
            trainedModel = TrainedModel(model.model, model.name, trainingMetrics.accuracy,
                validationMetrics.accuracy, trainingMetrics, validationMetrics)
            push!(trainedModels, trainedModel)
        end
        show(to)
        println()                
    end
    return trainedModels
end

"""Gets the relevant training and validation metrics obtained by cross-validating the model."""
function getCrossValidationMetrics(model::Model, scores::Any)
    trainingAccuracy = mean(scores["train_accuracy"])
    trainingMetrics = buildMetrics(
        accuracy=trainingAccuracy, 
        accuracySTD=std(scores["train_accuracy"]),
        errorRate=1-trainingAccuracy,
        fScore=mean(scores["train_f1_macro"]),
        fScoreSTD=std(scores["train_f1_macro"]),
        positivePredictiveValue=mean(scores["train_precision_macro"]),
        positivePredictiveValueSTD=std(scores["train_precision_macro"]),
        sensitivity=mean(scores["train_recall_macro"]),
        sensitivitySTD=std(scores["train_recall_macro"])
    )
    validationAccuracy = mean(scores["test_accuracy"])
    validationMetrics = buildMetrics(
        accuracy=mean(scores["test_accuracy"]), 
        accuracySTD=std(scores["test_accuracy"]),
        errorRate=1-validationAccuracy,
        fScore=mean(scores["test_f1_macro"]),
        fScoreSTD=std(scores["test_f1_macro"]),
        positivePredictiveValue=mean(scores["test_precision_macro"]),
        positivePredictiveValueSTD=std(scores["test_precision_macro"]),
        sensitivity=mean(scores["test_recall_macro"]),
        sensitivitySTD=std(scores["test_recall_macro"])
    )
    return trainingMetrics, validationMetrics
end

"""Prints the results of the evaluation of each model."""
function printResults(evaluations::Array{Evaluation})
    for eval=evaluations
        println(eval.model.name)
        println(eval.accuracy)
        println(eval.confMatrix)
        println(eval.evaluateMetrics)
        println(eval.evaluateMetrics.classificationReport)
    end
end
