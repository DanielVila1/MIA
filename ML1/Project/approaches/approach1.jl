
include("../utils/approachNew.jl")


# Functions to create the models for the approach.

function getAnns1()::Array{ANN}
    return [
        buildANN(layers=[32], maxIter=100),
        buildANN(layers=[16,32], maxIter=100),
        buildANN(layers=[16,8], maxIter=100),
        buildANN(layers=[4], maxIter=100),
        buildANN(layers=[8], maxIter=100),
        buildANN(layers=[16], maxIter=100),
        buildANN(layers=[8,8], maxIter=100),
        buildANN(layers=[16,16], maxIter=100),
    ]
end

function getSvms1()::Array{SVM}
    return [
        buildSVM(kernel="sigmoid", C=1.5),
        buildSVM(kernel="sigmoid", C=3.5),
        buildSVM(kernel="sigmoid", C=15.0, kernelGamma=2),
        buildSVM(kernel="sigmoid", C=20.0, kernelGamma=2),
        buildSVM(kernel="rbf", C=35.0, kernelGamma=2),
        buildSVM(kernel="rbf", C=50.0, kernelGamma=2),
        buildSVM(kernel="rbf", C=75.0, kernelGamma=2),
        buildSVM(kernel="rbf", C=100.0, kernelGamma=2),
    ]
end

function getDts1()::Array{DT}
    return [
        buildDT(depth=5),
        buildDT(depth=6),
        buildDT(depth=7),
        buildDT(depth=8),
        buildDT(depth=9),
        buildDT(depth=10),
    ]
end

function getkNNs1()::Array{KNN}
    return [
        buildKNN(neighbors=3),
        buildKNN(neighbors=4),
        buildKNN(neighbors=5),
        buildKNN(neighbors=6),
        buildKNN(neighbors=7),
        buildKNN(neighbors=8),
    ]
end


# Function to create the approach.

function createApproach1()
    approach = createApproach(
        name="approach1",
        multiplyClassSamples=[0.5,2,2,2,2], 
        anns=getAnns1(),
        svms=getSvms1(),
        dts=getDts1(),
        knns=getkNNs1(), 
        useKFoldCrossValidation=true,
        kFolds=5,
        ensembles=[SoftVoting, HardVoting, RandomForest],
        reduceDimensionality=true,
        dimensionalityReduction=DR_PCA
    )
    return approach
end
