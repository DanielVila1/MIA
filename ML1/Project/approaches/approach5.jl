
include("../utils/approach.jl")


# Functions to create the models for the approach.

function getAnns5()::Array{ANN}
    return [
        buildANN(layers=[4,4], maxIter=50),
        buildANN(layers=[8,8], maxIter=50),
        buildANN(layers=[2,2], maxIter=50),
        buildANN(layers=[16,8], maxIter=50),
        buildANN(layers=[4], maxIter=50),
        buildANN(layers=[8], maxIter=50),
        buildANN(layers=[16], maxIter=50),
        buildANN(layers=[32], maxIter=50),
    ]
end

function getSvms5()::Array{SVM}
    return []
end

function getDts5()::Array{DT}
    return [
        buildDT(depth=2),
        buildDT(depth=4),
        buildDT(depth=6),
        buildDT(depth=8),
        buildDT(depth=10),
        buildDT(depth=12),
    ]
end

function getkNNs5()::Array{KNN}
    return [
        buildKNN(neighbors=2),
        buildKNN(neighbors=4),
        buildKNN(neighbors=6),
        buildKNN(neighbors=8),
        buildKNN(neighbors=10),
        buildKNN(neighbors=12),
    ]
end


# Function to create the approach.

function createApproach5()
    approach = createApproach(
        name="approach5b",
        multiplyClassSamples=[0.5,5,5,5,5], 
        anns=getAnns5(),
        svms=getSvms5(),
        dts=getDts5(),
        knns=getkNNs5(), 
        useKFoldCrossValidation=false,
        validationRatio=0.2,
        ensembles=[HardVoting, GradientBoosting],
        reduceDimensionality=true,
        dimReduction=1.87,
        dimensionalityReduction=DR_PCA,
        customs=[buildCustom(binary=buildKNN(neighbors=6), multi=binary=buildKNN(neighbors=6))]
    )
    return approach
end
