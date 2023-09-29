
include("custom.jl")


# Type definitions.

"""Enum with all the possible classes of the multi-class problem."""
@enum Classes begin
    N = 0
    S = 1
    V = 2
    F = 3
    Q = 4
end

"""Represents an instance of the classification problem."""
struct Problem
    trainSetPath::String
    testSetPath::String
    isBinaryClassification::Bool
    negativeClasses::Array{Classes}
    positiveClasses::Array{Classes}
end

"""Creates an instance of the classification problem."""
function buildProblem(;trainSetPath::String, testSetPath::String, isBinaryClassification::Bool=false,
    negativeClasses::Array{Classes}=[N], positiveClasses::Array{Classes}=[S,V,F,Q]
)
    return Problem(trainSetPath, testSetPath, isBinaryClassification, negativeClasses, positiveClasses)
end

"""Represents an Artificial Neural Network architecture and hyper-parameters configuration."""
struct ANN
    layers::Array{Int}
    learningRateInit::Float64
    maxIter::Int32
    validationFraction::Float64
    solver::String
    activation::String
    learningRate::String
    nIterNoChange::Int32
end

"""Represents a Support-Vector Machine hyper-parameters configuration."""
struct SVM
    kernel::String
    kernelDegree::Int
    kernelGamma::Int
    C::Float64
    probability::Bool
    linear::Bool
    ensembleSoftVoting::Bool
    maxIter::Int
    bagged::Bool
end

"""Represents a Decision Tree hyper-parameters configuration."""
struct DT
    depth::Int
end

"""Represents a k-Nearest Neighbors hyper-parameters configuration."""
struct KNN
    neighbors::Int
    weights::String
end

"""Represents the different types of ensemble models supported."""
@enum EnsembleType begin
    HardVoting
    SoftVoting
    RandomForest
    GradientBoosting
    XGBoost
end

"""Represents an ensemble model configuration with its creator function."""
struct Ensemble
    type::EnsembleType
    needsModels::Bool
    creator::Any
end

struct Custom
    binary::Any
    multi::Any
end

"""Represents the specification of an approach to the classification problem."""
struct Approach
    name::String
    multiplyClassSamples::Array{Float64}
    anns::Array{ANN}
    svms::Array{SVM}
    dts::Array{DT}
    knns::Array{KNN}
    useKFoldCrossValidation::Bool
    kFolds::Int
    reduceDimensionality::Bool
    dimensionalityReduction::DimensionalityReduction
    ensembles::Array{EnsembleType}
    numberBestModels::Int
    validationRatio::Float64
    reportFolder::String
    showPlots::Bool
    dimReduction::Float64
    customs::Array{Custom}
    colsSelected::UnitRange{Int}  # used for custom dimReduction
end

"""Represents the metrics of interest for a certain trained model."""
struct Metrics
    accuracy::Float64
    accuracySTD::Float64
    errorRate::Float64
    fScore::Float64
    fScoreSTD::Float64
    negativePredictiveValue::Float64
    positivePredictiveValue::Float64
    positivePredictiveValueSTD::Float64
    sensitivity::Float64
    sensitivitySTD::Float64
    specificity::Float64
    classificationReport::String
end

Base.show(io::IO, m::Metrics) = print(io, "METRICS \n  - ACC = $(m.accuracy)\n  - PPV = $(m.positivePredictiveValue)\n  - REC = $(m.sensitivity)\n  - F1S = $(m.fScore)\n")
Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f)

"""Represents the evaluation metrics for a certain trained model."""
struct Evaluation
    model::Any
    accuracy::Float64
    confMatrix::Any
    evaluateMetrics::Metrics
end

"""Represents a classifier model."""
struct Model
    model::Any
    name::String
end

"""Represents a trained model."""
struct TrainedModel
    model::Any
    name::String
    trainingScore::Float64
    validationScore::Float64
    trainingMetrics::Metrics
    validationMetrics::Metrics
end

"""Represents a dataset, split in train and test sets.
The class column must be the last one."""
struct Dataset
    train::DataFrame
    trainProcessed::DataFrame
    trainOriginalSizes
    trainFinalSizes
    test::DataFrame
end

"""Represents a prepared dataset after being pre-processed, split in train and test sets,
and in inputs (the input feature vectors per sample) and targets (the real classes per sample).
The class column must be the last one."""
struct PreparedDataset
    trainInputs::Matrix
    trainTargets::Vector
    testInputs::Matrix
    testTargets::Vector
end


# Factory functions.

"""Builds a model specification with the given classifier model."""
function buildModel(;model=nothing, name::String="")
    return Model(model, name)
end

"""Creates all the models specified by the given approach."""
function createModels(approach::Approach)::Tuple{Array{Model},Array{Ensemble}}
    models::Array{Model} = []
    ensembles::Array{Ensemble} = []
    i = 0
    for ann=approach.anns
        name = "ANN_" * string(ann.layers)
        push!(models, buildModel(model=createANN(ann), name=name))
        i += 1
    end
    i = 0
    for svm=approach.svms
        if svm.linear
            name = string("SVM_linear_", svm.C)
        else
            name = "SVM_" * string(svm.kernel) * "_" * string(svm.C)
        end
        push!(models, buildModel(model=createSVM(svm), name=name))
        i += 1
    end
    i = 0
    for dt=approach.dts
        name = "DT_" * string(dt.depth)
        push!(models, buildModel(model=createDT(dt), name=name))
        i += 1
    end
    i = 0
    for knn=approach.knns
        name = "kNN_" * string(knn.neighbors, "_" , knn.weights)
        push!(models, buildModel(model=createKNN(knn), name=name))
        i += 1
    end
    i = 0
    for custom=approach.customs
        name = string("custom_", i)
        model = buildModel(model=createCustom(custom), name=name)
        push!(models, model)
        i += 1
    end
    for type=approach.ensembles
        ensemble = buildEnsemble(type)
        push!(ensembles, ensemble)
    end
    return models, ensembles
end

"""Creates an ANN specification with the given parameters."""
function buildANN(;
    layers::Array{Int}=[1],
    learningRateInit::Float64=0.01,
    maxIter::Int64=50,
    validationFraction::Float64=0.0,
    solver::String="adam",
    activation::String="logistic",
    learningRate::String="constant",
    nIterNoChange::Int64=50
    )
    return ANN(
        layers, learningRateInit, maxIter, validationFraction, solver, 
        activation, learningRate, nIterNoChange
    )    
end

"""Creates an k-Nearest Neighbors specification with the given parameters."""
function buildKNN(;neighbors::Int=4, weights::String="uniform")
    return KNN(neighbors, weights)
end

"""Creates an Decision Tree specification with the given parameters."""
function buildDT(;depth::Int=5)
    return DT(depth)
end

"""Creates an Support-Vector Machine specification with the given parameters."""
function buildSVM(;kernel::String="sigmoid", C::Float64=1.0, kernelDegree::Int=1,
    kernelGamma::Int=1, probability::Bool=false, linear::Bool=false, ensembleSoftVoting::Bool=false, 
    maxIter::Int=-1, bagged::Bool=false
)
    return SVM(kernel, kernelDegree, kernelGamma, C, probability,
            linear, ensembleSoftVoting, maxIter, bagged)
end

"""Creates a Metrics object with the given parameters."""
function buildMetrics(;
        accuracy::Float64=NaN,
        accuracySTD::Float64=NaN,
        errorRate::Float64=NaN,
        fScore::Float64=NaN,
        fScoreSTD::Float64=NaN,
        negativePredictiveValue::Float64=NaN,
        positivePredictiveValue::Float64=NaN,
        positivePredictiveValueSTD::Float64=NaN,
        sensitivity::Float64=NaN,
        sensitivitySTD::Float64=NaN,
        specificity::Float64=NaN,
        classificationReport::String=""
    )
    return Metrics(accuracy, accuracySTD, errorRate, fScore, fScoreSTD, 
        negativePredictiveValue, positivePredictiveValue, positivePredictiveValueSTD, sensitivity,
        sensitivitySTD, specificity, classificationReport
    )
end

"""Builds a custom classifier specification."""
function buildCustom(;binary=nothing, multi=nothing)
    return Custom(binary, multi)
end

"""Creates an ANN model from the given ANN specification."""
function createANN(ann::ANN)     
    return MLPClassifier(
        hidden_layer_sizes=ann.layers,
        max_iter=ann.maxIter,
        solver=ann.solver,
        activation=ann.activation,
        learning_rate=ann.learningRate,
        learning_rate_init=ann.learningRateInit,  
        n_iter_no_change=ann.nIterNoChange,
        validation_fraction=ann.validationFraction
    )
end

"""Creates a SVM model from the given SVM specification."""
function createSVM(svm::SVM)
    if svm.linear
        if svm.ensembleSoftVoting
            svmLinear = LinearSVC(
                C=svm.C,
                class_weight="balanced",
                max_iter=svm.maxIter
            )
            return CalibratedClassifierCV(svmLinear) 
        else
            return LinearSVC(
                C=svm.C,
                class_weight="balanced",
                max_iter=svm.maxIter
            )
        end
    else
        svc = SVC(kernel=svm.kernel,
            degree=svm.kernelDegree,
            gamma=svm.kernelGamma,
            probability=svm.probability,
            C=svm.C,
            class_weight="balanced", 
            max_iter=svm.maxIter
        )
        if svm.bagged
            return BaggingClassifier(base_estimator=svc, n_estimators=10, max_samples=0.1)
        else
            return svc
        end
    end
end

"""Creates a Decision Tree model from the given DT specification."""
function createDT(dt::DT)
    return DecisionTreeClassifier(max_depth=dt.depth)
end

"""Creates a kNN model from the given kNN specification."""
function createKNN(knn::KNN)
    return KNeighborsClassifier(knn.neighbors,weights=knn.weights)
end

"""Creates all the ensemble model specifications using the provided trained models
for the ensemble models that require base models."""
function createEnsembles(ensembles, trainedModels)::Array{Model}
    ensembleModels::Array{Model} = []
    baseModels = []
    i = 0
    for trainedModel=trainedModels
        baseModel = (string(trainedModel.name, "_", i), trainedModel.model)
        push!(baseModels, baseModel)
        i += 1
    end
    weights = map((m) -> m.validationScore, trainedModels)
    i = 0
    for ensemble=ensembles
        if ensemble.needsModels
            ensembleModel = ensemble.creator(baseModels, weights)
        else
            ensembleModel = ensemble.creator()
        end
        #= the name of the ensemble is the name of each model that belongs to it =#
        name=""
        if (ensemble.needsModels)
            name = string(ensemble.type) * ": [ "
            for model in baseModels
                name = name * ", "
                name = name *  string("", model[1],"")
            end
            name = name * " ]"
        else
            name = string(ensemble.type) * string(i)
        end
        
        push!(ensembleModels, Model(ensembleModel, name))
        i += 1
    end
    return ensembleModels
end

"""Creates an ensemble model specification for the given ensemble type."""
function buildEnsemble(type::EnsembleType)::Ensemble
    if type == HardVoting
        return Ensemble(type, true, createHardVotingEnsemble)
    elseif type == SoftVoting
        return Ensemble(type, true, createSoftVotingEnsemble)
    elseif type == RandomForest
        return Ensemble(type, false, createRandomForest)
    elseif type == GradientBoosting
        return Ensemble(type, false, createGradientBoosting)
    elseif type == XGBoost
        return Ensemble(type, false, createXGBoost)
    else
        return nothing
    end
end

"""Creates a classifier model based on the type specification."""
function createModel(spec::Any)
    if isa(spec, ANN)
        return createANN(spec)
    elseif isa(spec, SVM)
        return createSVM(spec)
    elseif isa(spec, DT)
        return createDT(spec)
    elseif isa(spec, KNN)
        return createKNN(spec)
    end
end

"""Creates an Custom model from the given specification."""
function createCustom(custom::Custom)
    binary = createModel(custom.binary)
    multi = createModel(custom.multi)
    cc = py"CustomClassifier"(binary, multi) 
    return cc
end
