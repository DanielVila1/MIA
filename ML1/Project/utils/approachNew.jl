
include("libraries.jl")
include("preprocess.jl")
include("dimensionality.jl")
include("ensembles.jl")
include("models.jl")
include("postProcessNew.jl")
include("approach.jl")

"""Runs a given approach over a given problem, training and evaluating all the models.""" 
function runAllPerModel(problem::Problem, approach::Approach)
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
    @timeit to "trainAndEvaluateAllModels" begin     
    models = trainAndEvaluateAllModels(problem, approach, normalized)
    end
    show(to)
    println()   
end


# Internal functions.

"""Trains and evaluates all the models for a given approach and problem with the normalized dataset.""" 
function trainAndEvaluateAllModels(problem::Problem, approach::Approach, normalized::PreparedDataset)
    models, ensembles = createModels(approach)
    results::Vector{Evaluation} = []
    trainedModels::Vector{TrainedModel} = []
    plotInfo = initializePlot(problem, approach)
    for model=models
        println(model.name)
        @timeit to "trainAndValidateModels" begin     
            trainedModel = trainAndValidateModel(approach, normalized, model)
        end
        show(to)
        println()
        push!(trainedModels, trainedModel)   
        @timeit to "evaluateModels" begin            
            result = evaluateModel(problem, approach, normalized, trainedModel)
        end
        show(to)
        println()
        push!(results, result)
        println("########################################################")
        println(trainedModel.name)
        println(trainedModel.validationMetrics.classificationReport)
        printResults(results)
        storeResultDataNew(approach.name, results; folder=approach.reportFolder)
        plotResult(plotInfo, problem, approach, result, normalized)
    end
    for ensemble=ensembles
        println(ensemble.type)
        @timeit to "trainAndValidateModels" begin     
            trainedEnsemble = trainAndValidateEnsemble(approach, normalized, ensemble, trainedModels)
        end
        show(to)
        println()   
        @timeit to "evaluateModels" begin            
            result = evaluateModel(problem, approach, normalized, trainedEnsemble)
        end
        show(to)
        println()
        push!(results, result)
        println("########################################################")
        println(trainedEnsemble.name)
        println(trainedEnsemble.validationMetrics.classificationReport)
        printResults(results)
        storeResultDataNew(approach.name, results; folder=approach.reportFolder)
        #plotResult(plotInfo, problem, approach, result, normalized)
    end
    showPlot(plotInfo, problem, approach)
    return results
end

"""Trains and validates a single model on the provided prepared dataset."""
function trainAndValidateModel(approach::Approach, dataset::PreparedDataset, model)
    trainedModels = trainModels(approach, dataset, [model])
    return trainedModels[1]
end

"""Trains and validates a single ensemble (with the provided base models if needed) on the provided prepared dataset."""
function trainAndValidateEnsemble(approach::Approach, dataset::PreparedDataset, ensemble, trainedModels)
    trainedEnsembles = trainEnsembles(approach, dataset, [ensemble], trainedModels)
    return trainedEnsembles[1]
end

"""Evaluates a single trained model on the provided on the test set."""
function evaluateModel(problem::Problem, approach::Approach, dataset::PreparedDataset, trainedModel::TrainedModel)::Evaluation
    evaluations = evaluateModels(problem, approach, dataset, [trainedModel])           
    return evaluations[1]
end

"""Checks if a plot can be printed for the given problem/approach."""
function canPrintPlot(problem::Problem, approach::Approach)
    return problem.isBinaryClassification && approach.showPlots
end

"""Initializes a plot for the given problem/approach."""
function initializePlot(problem::Problem, approach::Approach)
    if !canPrintPlot(problem, approach) return nothing end
    return PyPlot.gca()
end

"""Show a plot for the given problem/approach."""
function showPlot(plotInfo, problem::Problem, approach::Approach)
    if !canPrintPlot(problem, approach) return end
    show()
end

"""Plot a ROC curve for the provided evaluation of a model and for the given problem/approach
on the initialized plot (plotInfo)."""
function plotResult(plotInfo, problem::Problem, approach::Approach, evaluation::Evaluation, dataset::PreparedDataset)
    if !canPrintPlot(problem, approach) return end
    RocCurveDisplay.from_estimator(evaluation.model.model, dataset.testInputs, dataset.testTargets, ax=plotInfo)
end
