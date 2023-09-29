
# Julia packages.

using Downloads
using DelimitedFiles
using Plots
using UrlDownload
using DataFrames
using StatsBase
using Random
using ScikitLearn
using ScikitLearn: fit!
using ScikitLearn: score
using ScikitLearn: predict
using ScikitLearn.Models
using CSV
using TimerOutputs
using PyCall
using Printf
using PyPlot


# Scikit-Learn imports.

@sk_import metrics: RocCurveDisplay
@sk_import ensemble:VotingClassifier
@sk_import ensemble:GradientBoostingClassifier
@sk_import ensemble: RandomForestClassifier
@sk_import svm:SVC
@sk_import svm:LinearSVC
@sk_import calibration:CalibratedClassifierCV 
@sk_import tree:DecisionTreeClassifier
@sk_import linear_model:LogisticRegression
@sk_import naive_bayes:GaussianNB 
@sk_import neighbors: KNeighborsClassifier
@sk_import neural_network: MLPClassifier
@sk_import ensemble: GradientBoostingClassifier
@sk_import metrics: (accuracy_score, f1_score, precision_score, recall_score, classification_report)
@sk_import model_selection: (cross_validate, train_test_split)
@sk_import ensemble:BaggingClassifier
@sk_import discriminant_analysis: LinearDiscriminantAnalysis
@sk_import manifold:TSNE
@sk_import decomposition:FastICA
@sk_import decomposition:PCA


# Anaconda/Python imports.

xgb = pyimport_conda("xgboost", "xgboost")

np = pyimport("numpy")


# Fix random seed.

RANDOM_SEED = 1234
Random.seed!(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
# https://scikit-learn.org/stable/common_pitfalls.html#randomness


################ start one hot enconding functions #######################

function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    return permutedims(classes) .== feature
end

function oneHotEncoding(feature::AbstractArray{<:Any,1})
    oneHotEncoding(feature, unique(feature));
end

function oneHotEncoding(feature::AbstractArray{Bool,1})
    oneHotEncoding(feature, unique(feature));
end

function oneHotDecoding(outputs::AbstractArray{<:Real,2}) 
    if size(outputs, 2) == 1
        return outputs
    end
    (_, indicesMaxEachInstance) = findmax(outputs, dims=2);
    classes = [i[2] for i in indicesMaxEachInstance]
    if size(outputs, 2) == 2
        classes = classes .== 2
    end
    return classes[:]
end

################ end    one hot enconding functions #######################

################ start normalize functions #######################
function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return (minimum(dataset, dims=1), maximum(dataset, dims=1));
end

function normalizeMinMax!(dataset::AbstractArray{<:Real,2},      
        normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    dataset = dataset.-normalizationParameters[1];
    R12 = normalizationParameters[2]-normalizationParameters[1];
    dataset = dataset./R12;
    dataset = map(x -> isnan(x) ? zero(x) : x, dataset); 
end   

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    normalizeMinMax!(dataset, calculateMinMaxNormalizationParameters(dataset))
end

function normalizeMinMax( dataset::AbstractArray{<:Real,2},      
                normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) 
    dataSetCopy = copy(dataset);
    normalizeMinMax!(dataSetCopy, normalizationParameters)
    return dataSetCopy;
end

function normalizeMinMax( dataset::AbstractArray{<:Real,2})
    dataSetCopy = copy(dataset);
    normalizeMinMax!(dataSetCopy,calculateMinMaxNormalizationParameters(dataSetCopy));
    return dataSetCopy;    
end

function calculateZeroMeanNormalizationParameters(_dataToZeroMeanNorm::AbstractArray{<:Real,2})
    return (mean(_dataToZeroMeanNorm, dims=1), std(_dataToZeroMeanNorm, dims=1));
end

function normalizeZeroMean!(data_to_normalize::AbstractArray{<:Real,2},
            normalization_parameters::NTuple{2, AbstractArray{<:Real,2}}) 
    data_to_normalize = data_to_normalize .- normalization_parameters[1];
    data_to_normalize = data_to_normalize ./ normalization_parameters[2];
    data_to_normalize = map(x -> isnan(x) ? zero(x) : x, data_to_normalize);    
end                

function normalizeZeroMean!(data_to_normalize::AbstractArray{<:Real,2})
    normalizeZeroMean!(data_to_normalize, calculateZeroMeanNormalizationParameters(data_to_normalize));
end 

################ end normalize functions #######################

################ start hold out functions #######################

function classifyOutputs(outputs::AbstractArray{<:Real,2}; 
    threshold::Real=0.5) 
    if size(outputs, 2) == 1
        return outputs .>= threshold
    else 
        (_,indicesMaxEachInstance) = findmax(outputs, dims=2);
        outputs = falses(size(outputs));
        outputs[indicesMaxEachInstance] .= true;
        return outputs
    end
end

function holdout(num_samples::Int, ratio::Real)
    rp = randperm(num_samples)
    last_index = trunc(Int, num_samples * ratio)
    return (rp[1:last_index], rp[(last_index+1):end])
end

function holdout(num_samples::Int, validation_ratio::Real, test_ratio::Real) 
    @assert (validation_ratio + test_ratio <= 1.0)
    hold_out_validation = holdout(num_samples, validation_ratio)
    validation = hold_out_validation[1]
    rest = hold_out_validation[2]
    test_ratio_scaled = test_ratio / (1-validation_ratio)
    holdout_rest = holdout(size(rest,1), test_ratio_scaled)
    testing = rest[holdout_rest[1]]
    training = rest[holdout_rest[2]]
    return (training, validation, testing)
end

function holdout_normalized!(inputs, outputs, validation_ratio::Real, test_ratio::Real)
    normalizeZeroMean!(input_data)
    train_indexes, validation_indexes, test_indexes = holdout(size(input_data,1), validation_ratio, test_ratio)
    return (inputs[train_indexes,:], outputs[train_indexes], inputs[test_indexes,:], outputs[test_indexes])
end

################ end hold out functions #######################

################ start cross validation functions ###################

function crossvalidation(N::Int64, k::Int64)
    index = collect(1:k)
    repetitions = repeat(index, Int(ceil(N/k)))
    indexSplits = repetitions[1:N]
    indexSplits = shuffle!(indexSplits)
    return indexSplits
end

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    nExamples = size(targets, 1)
    nClasses = size(targets, 2)
    indexes = Array((1:nExamples)) 
    for i=1:nClasses
        nElementsClassI = sum(targets[:,i])
        classIndexes = findall(>(0), targets[:,i])
        cv = crossvalidation(nElementsClassI, k)
        indexes[classIndexes] = cv
    end 
    return indexes
end

function crossvalidationWithoutOneHot(targets::AbstractArray{<:Any,1}, k::Int64)
    nExamples = size(targets,1)
    classes = unique(targets)
    nClasses = size(classes,1)
    indexes = Array((1:nExamples)) 
    for i=1:nClasses
        classI = classes[i]
        classIndexes = findall(==(classI), targets)
        nElementsClassI = size(classIndexes,1)
        cv = crossvalidation(nElementsClassI, k)
        indexes[classIndexes] = cv
    end 
    return indexes
end

# symbol = :ANN, :SVM, :DecisionTree or :kNN
#In turn, this function should return, at least, the values for the selected metrics.
function modelCrossValidation(modelType::Symbol,
    modelHyperparameters::Dict,
    inputs::AbstractArray{<:Real,2},
    targets::AbstractArray{<:Any,1},
    crossValidationIndices::Array{Int64,1})
# cross validation tasks
k = maximum(crossValidationIndices)
vecAccuracy = zeros(Float64, k)
vecErrRate = zeros(Float64, k)
vecSensivity = zeros(Float64, k)
vecSpecifity = zeros(Float64, k)
vecPositivePred = zeros(Float64, k)
vecNegativePred = zeros(Float64, k)
vecHarmonicMean = zeros(Float64, k)

if(modelType==:ANN)
   # targets = oneHotEncoding(targets, unique(targets))
end
# cross validation loop k folds
for i=1:k
    testInput = inputs[findall(==(i), crossValidationIndices),:]
    testOutput = targets[findall(==(i), crossValidationIndices)]
    trainingInput = inputs[findall(!=(i), crossValidationIndices),:]
    trainingOutput = targets[findall(!=(i), crossValidationIndices)]
    local model 
    if(modelType==:ANN)
        @assert(haskey(modelHyperparameters,"AnnArch"))
        @assert(haskey(modelHyperparameters,"AnnLr"))
        @assert(haskey(modelHyperparameters,"AnnRatioPatters"))
        @assert(haskey(modelHyperparameters,"AnnMaxEpochWithoutImpr"))
        @assert(haskey(modelHyperparameters,"NumberEpochs")) 
        trainingOutput = oneHotEncoding(trainingOutput, unique(trainingOutput))
        model = MLPClassifier(hidden_layer_sizes=modelHyperparameters["AnnArch"], 
            max_iter=modelHyperparameters["NumberEpochs"],
            solver="adam",
            activation="logistic",
            learning_rate="constant",
            random_state=10,
            learning_rate_init=modelHyperparameters["AnnLr"],  
            n_iter_no_change=modelHyperparameters["AnnMaxEpochWithoutImpr"],
            validation_fraction=modelHyperparameters["AnnRatioPatters"]
        )            
        fit!(model, trainingInput,trainingOutput)
        outputs = predict(model,testInput)     
        confMat = confusionMatrix(convert(Matrix{Bool}, outputs), oneHotEncoding(testOutput, unique(testOutput)), weighted=false) 
    else
        if (modelType==:SVC)
            @assert(haskey(modelHyperparameters,"kernel"))
            @assert(haskey(modelHyperparameters,"kernelDegree"))
            @assert(haskey(modelHyperparameters,"kernelGamma"))
            @assert(haskey(modelHyperparameters,"C"))
            model = SVC(kernel=modelHyperparameters["kernel"], 
                degree=modelHyperparameters["kernelDegree"], 
                gamma=modelHyperparameters["kernelGamma"], 
                C=modelHyperparameters["C"]);  
            fit!(model, trainingInput,trainingOutput)
            outputs = predict(model,testInput)           
            confMat = confusionMatrix(outputs, testOutput, weighted=false)  
        end
        if (modelType==:DecisionTree)
            @assert(haskey(modelHyperparameters,"max_depth"))
            @assert(haskey(modelHyperparameters,"random_state"))            
            model = DecisionTreeClassifier(max_depth=modelHyperparameters["max_depth"],
                random_state=modelHyperparameters["random_state"]);
            fit!(model, trainingInput,trainingOutput)
            outputs = predict(model,testInput)           
            confMat = confusionMatrix(outputs, testOutput, weighted=false)                  
        end
        if (modelType==:kNN)
            @assert(haskey(modelHyperparameters,"kNN"))    
            model = KNeighborsClassifier(modelHyperparameters["kNN"]); 
            fit!(model, trainingInput,trainingOutput)
            outputs = predict(model,testInput)           
            confMat = confusionMatrix(outputs, testOutput, weighted=false)                
        end 
        if (modelType==:NB)  
            model = GaussianNB(); 
            fit!(model, trainingInput,trainingOutput)
            outputs = predict(model,testInput)           
            confMat = confusionMatrix(outputs, testOutput, weighted=false)                
        end  
        if (modelType==:LR)  
            model = LogisticRegression(); 
            fit!(model, trainingInput,trainingOutput)
            outputs = predict(model,testInput)           
            confMat = confusionMatrix(outputs, testOutput, weighted=false)                
        end                 
    end
    vecAccuracy[i] = confMat.accuracy;
    vecErrRate[i] = confMat.errorRate
    vecSensivity[i] = confMat.sensitivity;
    vecSpecifity[i] = confMat.specificity;
    vecPositivePred[i] = confMat.ppv;
    vecNegativePred[i] = confMat.npv;
    vecHarmonicMean[i] = confMat.fScore;        
end # end kfold loop

return (
    meanAccuracy = mean(vecAccuracy),
    errorRate = mean(vecErrRate),
    stdAccuracy = std(vecAccuracy),
    meanSensitivity = mean(vecSensivity),
    stdSensitivity = std(vecSensivity),
    meanSpecifity = mean(vecSpecifity),
    stdSpecifity = std(vecSpecifity),
    meanPp = mean(vecPositivePred),
    stdPp = std(vecPositivePred),
    meanNp = mean(vecNegativePred),
    stdNp = std(vecNegativePred),
    meanHarmo = mean(vecHarmonicMean),
    stdHarmo = std(vecHarmonicMean),
)
end

################ end cross validation functions ###################

################ start create model functions ###########################

function createModel(modelType::Symbol, hyperparams::Dict{String,Any})
    if modelType == :ANN
        return MLPClassifier(
            hidden_layer_sizes=hyperparams["AnnArch"], 
            max_iter=hyperparams["Epochs"],
            solver="adam",
            activation="logistic",
            learning_rate="constant",
            random_state=10,
            learning_rate_init=hyperparams["AnnLR"],  
            n_iter_no_change=hyperparams["AnnMaxEpochsVal"],
            validation_fraction=hyperparams["AnnValidationRatio"]
        )            
    elseif modelType == :SVC
        return SVC(
            kernel=hyperparams["kernel"], 
            degree=hyperparams["kernelDegree"], 
            gamma=hyperparams["kernelGamma"], 
            C=hyperparams["C"],
            probability=true
        )          
    elseif modelType == :DecisionTree
        return DecisionTreeClassifier(
            max_depth=hyperparams["max_depth"],
            random_state=hyperparams["random_state"]
        )
    elseif modelType == :kNN
        return KNeighborsClassifier(hyperparams["kNN"])
    else
        throw(ArgumentError("Model type \"$(modelType)\" not supported!"))
    end 
end

################ end create model functions ###########################

################ start train class ensemble ###########################

function trainClassEnsemble(estimators::AbstractArray{Symbol,1}, 
    modelsHyperParameters::AbstractArray{Dict{String,Any}, 1},     
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},    
    kFoldIndices::Array{Int64,1};
    hardVoting::Bool=true
)
# trainClassEnsemble
k = maximum(kFoldIndices)
result_metrics = zeros(Float64, k)
global ensemble = nothing
for i=1:k
    testInputs = trainingDataset[1][findall(==(i), kFoldIndices),:]
    testTargets = trainingDataset[2][findall(==(i), kFoldIndices),:]
    trainInputs = trainingDataset[1][findall(!=(i), kFoldIndices),:]
    trainTargets = trainingDataset[2][findall(!=(i), kFoldIndices),:]
    baseModels = []
    numEstimators = length(estimators)
    weights = zeros(numEstimators)
    for j=1:numEstimators
        estimator = estimators[j]
        model = createModel(estimator, modelsHyperParameters[j])
        fit!(model, trainInputs, vec(trainTargets))
        push!(baseModels, (String(estimator)*string(j), model))
        if !hardVoting
            modelScore = score(model, trainInputs, vec(trainTargets))
            weights[j] = modelScore
        end
    end
    if hardVoting
        ensemble = VotingClassifier(estimators=baseModels, n_jobs=1, voting="hard")
    else
        ensemble = VotingClassifier(estimators=baseModels, n_jobs=1, voting="soft", weights=weights)
    end
    fit!(ensemble, trainInputs, vec(trainTargets))
    result_metrics[i] = score(ensemble, testInputs, vec(testTargets))
end
fit!(ensemble, trainingDataset[1], vec(trainingDataset[2]))
return (model=ensemble, avg=mean(result_metrics), std=std(result_metrics))
end

function trainClassEnsemble(baseEstimator::Symbol, 
modelsHyperParameters::Dict{String,Any},
trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},     
kFoldIndices::Array{Int64,1};
numEstimators::Int=100,
hardVoting::Bool=true
)
# trainClassEnsemble
estimators = [baseEstimator for i=1:numEstimators]
hyperparameters = [modelsHyperParameters for i=1:numEstimators]
return trainClassEnsemble(estimators, hyperparameters, trainingDataset, kFoldIndices, hardVoting=hardVoting)
end

function printEnsembleResults(results)
println("AVG = $(results.avg) (STD = $(results.std))")
end

################ end train class ensemble ######################


################ start confusion matrix ########################

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) 
    return mean(outputs .== targets)
end


function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}) 
    if size(outputs, 2) == 1
        return accuracy(outputs[:,1], targets[:,1])
    else
        return mean(all(outputs .== targets, dims=2))
    end
end

@enum CONFUSION begin
    TP = 1
    TN = 2
    FP = 3
    FN = 4
end

function calculatePattern(output::Bool, target::Bool)::CONFUSION
    if (output && target)
        return TP
    end
    if (!output && !target)
        return TN
    end
    if (output && !target)
        return FP
    end   
    if (!output && target)
        return FN
    end     
end

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    confMatrix = Array{Int64}(undef,2,2)
    # perform metrics calc
    tnValue = sum(calculatePattern.(outputs, targets).==TN)
    fpValue = sum(calculatePattern.(outputs, targets).==FP)
    fnValue = sum(calculatePattern.(outputs, targets).==FN)
    tpValue = sum(calculatePattern.(outputs, targets).==TP)
    confMatrix = [tnValue fpValue; fnValue tpValue]
    # check if every pattern is TP or TN
    targetLength = length(targets)
    allTn = tnValue == targetLength
    allTp = tpValue == targetLength
    sumAll = tnValue + tpValue + fpValue + fnValue
    @assert (targetLength == sumAll)
    # Accuracy = 𝑇𝑁+𝑇𝑃 / 𝑇𝑁+𝑇𝑃+𝐹𝑁+𝐹𝑃
    accuracy = (tnValue + tpValue) / sumAll
    # Error rate = 𝐹𝑃+𝐹𝑁 / 𝑇𝑁+𝑇𝑃+𝐹𝑁+𝐹𝑃
    errorRate = (fpValue + fnValue) / sumAll
    @assert (isapprox(errorRate, (1.0 - accuracy)))
    # Sensitivity = 𝑇𝑃 / 𝐹𝑁+𝑇𝑃
    sensitivity = allTn ? 1 : 0
    if (tpValue + fnValue) > 0
        sensitivity = tpValue / (tpValue + fnValue)
    end   
    # Specificity = 𝑇𝑁 / 𝐹𝑃+𝑇𝑁
    specificity = allTp ? 1 : 0
    if (fpValue + tnValue) > 0
        specificity = tnValue / (fpValue + tnValue)
    end
    # Positive predicitive value = 𝑇𝑃 / 𝑇𝑃+𝐹𝑃
    positivePred = allTn ? 1 : 0
    if (tpValue + fpValue) > 0
        positivePred = tpValue / (tpValue + fpValue)
    end
    # Negative predicitve value = 𝑇𝑁 / 𝑇𝑁+𝐹𝑁
    negativePred = allTp ? 1 : 0
    if (fnValue + tnValue) > 0
        negativePred = tnValue / (fnValue + tnValue)
    end
    # F-score = harmonic mean of precision and recall
    fScore = 0
    if (sensitivity > 0 || positivePred > 0)
        fScore = harmmean([positivePred, sensitivity])
        #@assert(fScore == ((2 * positivePred * sensitivity) / (positivePred + sensitivity)))
    end
    tupleResults = (accuracy = accuracy, errorRate = errorRate, sensitivity = sensitivity, 
        specificity = specificity, ppv = positivePred, npv = negativePred, fScore = fScore,
        confusionMatrix = confMatrix)
    return tupleResults
end

function confusionMatrix(outputs::AbstractArray{<:Real,1},targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    # convert the outputs in boolean using threshold
    outputsBool = outputs .>= threshold;
    # call previous function
    return confusionMatrix(outputsBool, targets)
end

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=false)
    classifiedOutputs = classifyOutputs(outputs)
    return confusionMatrix(classifiedOutputs, targets, weighted=weighted)
end

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=false)
    @assert (size(outputs) == size(targets))
    @assert (issubset(unique(outputs), unique(targets)))
    classes = unique(targets)
    outputsEncoded = oneHotEncoding(outputs, classes)
    targetsEncoded = oneHotEncoding(targets, classes)
    return confusionMatrix(outputsEncoded, targetsEncoded, weighted=weighted)
end

# From previous notebooks.
function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5) 
    if size(outputs, 2) == 1
        return outputs .>= threshold
    else 
        (_,indicesMaxEachInstance) = findmax(outputs, dims=2);
        outputs = falses(size(outputs));
        outputs[indicesMaxEachInstance] .= true;
        return outputs
    end
end

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=false)
    @assert (size(outputs) == size(targets))
    numClass = size(outputs, 2)
    if numClass == 2
        throw(DomainError("For 2 classes use boolean vector."))
    end
    if numClass == 1
        outputsVector = vec(outputs)
        targetsVector = vec(targets)
        return confusionMatrix(outputsVector, targetsVector)
    end
    sensitivity = zeros(numClass)
    errRate = zeros(numClass)
    specificity = zeros(numClass)
    ppv = zeros(numClass)
    npv = zeros(numClass)
    fScore = zeros(numClass)
    for i in 1:numClass
        cm = confusionMatrix(outputs[:,i], targets[:,i])
        errRate[i] = cm.errorRate
        sensitivity[i] = cm.sensitivity
        specificity[i] = cm.specificity
        ppv[i] = cm.ppv
        npv[i] = cm.npv
        fScore[i] = cm.fScore
    end
    confusion = zeros(numClass, numClass)
    for i in 1:numClass
        for j in 1:numClass
            confusion[i,j] = round.(sum(targets[:,i] .&& outputs[:,j]))
        end
    end
    if weighted
        sumClasses = sum(targets, dims=2)
        sumTotal = sum(sumClasses)
        weights = sumClasses ./ sumTotal
        sensitivity = sum(sensitivity .* weights)
        specificity = sum(specificity .* weights)
        ppv = sum(ppv .* weights)
        npv = sum(npv .* weights)
        fScore = sum(fScore .* weights)
        errRate = mean(errRate)
    else
        sensitivity = mean(sensitivity)
        specificity = mean(specificity)
        ppv = mean(ppv)
        npv = mean(npv)
        fScore = mean(fScore)
        errRate = mean(errRate)
    end
    acc = accuracy(outputs, targets)
    return (accuracy = acc, errorRate =errRate, sensitivity = sensitivity, specificity = specificity,
        ppv = ppv, npv = npv, fScore = fScore, confusionMatrix = confusion)
end

################ end confusion matrix ##########################

function createModel(modelType::Symbol, hyperparams::Dict{String,Any})
    if modelType == :ANN
        return MLPClassifier(
            hidden_layer_sizes=hyperparams["AnnArch"], 
            max_iter=hyperparams["Epochs"],
            solver="adam",
            activation="logistic",
            learning_rate="constant",
            random_state=10,
            learning_rate_init=hyperparams["AnnLR"],  
            n_iter_no_change=hyperparams["AnnMaxEpochsVal"],
            validation_fraction=hyperparams["AnnValidationRatio"]
        )            
    elseif modelType == :SVC
        return SVC(
            kernel=hyperparams["kernel"], 
            degree=hyperparams["kernelDegree"], 
            gamma=hyperparams["kernelGamma"], 
            C=hyperparams["C"],
            probability=true
        )          
    elseif modelType == :DecisionTree
        return DecisionTreeClassifier(
            max_depth=hyperparams["max_depth"],
            random_state=hyperparams["random_state"]
        )
    elseif modelType == :kNN
        return KNeighborsClassifier(hyperparams["kNN"])
    else
        throw(ArgumentError("Model type \"$(modelType)\" not supported!"))
    end 
end

function trainClassEnsemble(estimators::AbstractArray{Symbol,1}, 
        modelsHyperParameters::AbstractArray{Dict{String,Any}, 1},     
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},    
        kFoldIndices::Array{Int64,1};
        hardVoting::Bool=true
    )
    # trainClassEnsemble
    k = maximum(kFoldIndices)
    result_metrics = zeros(Float64, k)
    global ensemble = nothing
    for i=1:k
        testInputs = trainingDataset[1][findall(==(i), kFoldIndices),:]
        testTargets = trainingDataset[2][findall(==(i), kFoldIndices),:]
        trainInputs = trainingDataset[1][findall(!=(i), kFoldIndices),:]
        trainTargets = trainingDataset[2][findall(!=(i), kFoldIndices),:]
        baseModels = []
        numEstimators = length(estimators)
        weights = zeros(numEstimators)
        for j=1:numEstimators
            estimator = estimators[j]
            model = createModel(estimator, modelsHyperParameters[j])
            fit!(model, trainInputs, vec(trainTargets))
            push!(baseModels, (String(estimator)*string(j), model))
            if !hardVoting
                modelScore = score(model, trainInputs, vec(trainTargets))
                weights[j] = modelScore
            end
        end
        if hardVoting
            ensemble = VotingClassifier(estimators=baseModels, n_jobs=1, voting="hard")
        else
            ensemble = VotingClassifier(estimators=baseModels, n_jobs=1, voting="soft", weights=weights)
        end
        fit!(ensemble, trainInputs, vec(trainTargets))
        result_metrics[i] = score(ensemble, testInputs, vec(testTargets))
    end
    fit!(ensemble, trainingDataset[1], vec(trainingDataset[2]))
    return (model=ensemble, avg=mean(result_metrics), std=std(result_metrics))
end

function trainClassEnsemble(baseEstimator::Symbol, 
    modelsHyperParameters::Dict{String,Any},
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},     
    kFoldIndices::Array{Int64,1};
    numEstimators::Int=100,
    hardVoting::Bool=true
)
    # trainClassEnsemble
    estimators = [baseEstimator for i=1:numEstimators]
    hyperparameters = [modelsHyperParameters for i=1:numEstimators]
    return trainClassEnsemble(estimators, hyperparameters, trainingDataset, kFoldIndices, hardVoting=hardVoting)
end

function printEnsembleResults(results)
    println("AVG = $(results.avg) (STD = $(results.std))")
end
