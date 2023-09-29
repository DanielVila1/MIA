
# Enum of the available dimensionality reduction techniques.

@enum DimensionalityReduction begin
    DR_PCA
    DR_ICA
    DR_LDA
    DR_TSNE
    CUSTOM
end


# Main functions.

"""Applies CUSTOM reduction for choose only specific columns."""
function reducesDimensionalityCustom(colsSelected::UnitRange{Int},trainSet, testSet)
    println("Start reducing dimensionalty using CUSTOM")
    newTrain=select(trainSet, colsSelected, 188)
    newTest=select(testSet, colsSelected, 188)
    println("Train Patterns ", size(newTrain), " -> ", size(newTrain))
    println("Test Patterns ", size(newTest), " -> ", size(newTest))                    
    return (train=newTrain, test=newTest) 
end

"""Applies the specified dimensionality reduction technique to the given train and test sets."""
function reduceDimensionality(type::DimensionalityReduction, trainSet::DataFrame, testSet::DataFrame; dimReduction::Float64=2.0
        )::NamedTuple{(:train, :test), Tuple{DataFrame, DataFrame}}
    trainInputs = Matrix(trainSet[:,1:end-1])
    trainTargets = Vector(trainSet[:,end])
    testInputs = Matrix(testSet[:,1:end-1])
    testTargets = Vector(testSet[:,end])
    components = trunc(Int, size(trainInputs)[2]/dimReduction)
    if type == DR_PCA
        trainInputs, testInputs = reduceDimensionalityPCA(components, trainInputs, testInputs)
    elseif type == DR_ICA
        trainInputs, testInputs = reduceDimensionalityICA(components, trainInputs, testInputs)
    elseif type == DR_LDA
        numClasses = length(unique(trainSet[:,end])) 
        ldaComponents = min(components, numClasses-1)
        trainInputs, testInputs = reduceDimensionalityLDA(ldaComponents, trainInputs, trainTargets, testInputs)
    elseif type == DR_TSNE
        trainInputs, testInputs = reduceDimensionalityTSNE(components, trainInputs, testInputs)
    end
    reducedTrain = DataFrame(hcat(trainInputs, trainTargets), :auto)
    reducedTest = DataFrame(hcat(testInputs, testTargets), :auto)
    return (train=reducedTrain, test=reducedTest)
end


# Internal functions.

"""Applies the PCA dimensionality reduction to the given train and test sets."""
function reduceDimensionalityPCA(components::Int, trainSet, testSet)
    println("Start reducing dimensionalty using PCA")
    pca = PCA(components)
    fit!(pca, trainSet)
    pcaTrain = pca.transform(trainSet)
    pcaTest = pca.transform(testSet)
    println("Train Patterns ", size(trainSet), " -> ", size(pcaTrain))
    println("Test Patterns ", size(testSet), " -> ", size(pcaTest))
    return (train=pcaTrain, test=pcaTest)
end

"""Applies the ICA dimensionality reduction to the given train and test sets."""
function reduceDimensionalityICA(components::Int, trainSet, testSet)
    println("Start reducing dimensionalty using ICA")
    ica = FastICA(n_components=components)
    fit!(ica, trainSet)
    icaTrain = ica.transform(trainSet)
    icaTest = ica.transform(testSet)
    println("Train Patterns ", size(trainSet), " -> ", size(icaTrain))
    println("Test Patterns ", size(testSet), " -> ", size(icaTest))
    return (train=icaTrain, test=icaTest)    
end

"""Applies the LDA dimensionality reduction to the given train and test sets."""
function reduceDimensionalityLDA(components::Int, trainSet, trainTargets, testSet)
    println("Start reducing dimensionalty using LDA")
    lda = LinearDiscriminantAnalysis(n_components=components)
    fit!(lda, trainSet, trainTargets)
    ldaTrain = lda.transform(trainSet)
    ldaTest = lda.transform(testSet)
    println("Train Patterns ", size(trainSet), " -> ", size(ldaTrain))
    println("Test Patterns ", size(testSet), " -> ", size(ldaTest))    
    return (train=ldaTrain, test=ldaTest)
end

"""Applies the TSNE dimensionality reduction to the given train and test sets."""
function reduceDimensionalityTSNE(components::Int, trainSet, testSet)
    println("Start reducing dimensionalty using TSNE")
    tsne = TSNE(n_components=components, perplexity=10)
    tsneTrain = tsne.fit_transform(trainSet) 
    tsneTest = tsne.fit_transform(testSet)
    println("Train Patterns ", size(trainSet), " -> ", size(tsneTrain))
    println("Test Patterns ", size(testSet), " -> ", size(tsneTest))      
    return (train=tsneTrain, test=tsneTest)
end
