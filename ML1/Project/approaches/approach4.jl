
include("../utils/approachNew.jl")


# Functions to create the models for the approach.

function getAnns4()::Array{ANN}
    return [
        buildANN(layers=[4],maxIter=400),
        buildANN(layers=[8],maxIter=400),
        buildANN(layers=[16],maxIter=400),
        buildANN(layers=[6],maxIter=400),
        buildANN(layers=[8,8],maxIter=400),
        buildANN(layers=[4,4],maxIter=400),
        buildANN(layers=[2,2],maxIter=400),
        buildANN(layers=[16,8],maxIter=400),
    ]
end

function getSvms4()::Array{SVM}
    return [
        buildSVM(kernel="sigmoid", C=1.5, maxIter=300, linear=false, probability=true, ensembleSoftVoting=false),
        buildSVM(kernel="sigmoid", C=2.0, maxIter=300, linear=false, probability=true, ensembleSoftVoting=false),
        buildSVM(kernel="sigmoid", C=1.5, kernelGamma=2, maxIter=300, linear=false, probability=true, ensembleSoftVoting=false),
        buildSVM(kernel="sigmoid", C=2.0, kernelGamma=2, maxIter=300, linear=false, probability=true, ensembleSoftVoting=false),
        buildSVM(kernel="rbf", C=1.5, maxIter=300, linear=false, probability=true, ensembleSoftVoting=false),
        buildSVM(kernel="rbf", C=2.0, maxIter=300, linear=false, probability=true, ensembleSoftVoting=false),
        buildSVM(kernel="rbf", C=1.5, kernelGamma=2, maxIter=300, linear=false, probability=true, ensembleSoftVoting=false),
        buildSVM(kernel="rbf", C=2.0, kernelGamma=2, maxIter=300, linear=false, probability=true, ensembleSoftVoting=false)     
    ]
end

function getDts4()::Array{DT}
    return [
        buildDT(depth=3),
        buildDT(depth=4),
        buildDT(depth=5),
        buildDT(depth=6),
        buildDT(depth=7),
        buildDT(depth=8),
    ]
end

function getkNNs4()::Array{KNN}
    return [
        buildKNN(neighbors=4),
        buildKNN(neighbors=4, weights="distance"),
        buildKNN(neighbors=5),
        buildKNN(neighbors=5, weights="distance"),
        buildKNN(neighbors=6),
        buildKNN(neighbors=6, weights="distance"),
    ]
end

# Function to create the approach.

function createApproach4()
    approach = createApproach(
        name="approach4",
        multiplyClassSamples=[0.5,1,1,1,1],
        anns=getAnns4(),
        svms=getSvms4(),
        dts=getDts4(),
        knns=getkNNs4(), 
        useKFoldCrossValidation=true,
        kFolds=10,
        ensembles=[SoftVoting, XGBoost],
        numberBestModels=4,
        reduceDimensionality=true,
        dimensionalityReduction=CUSTOM,        
        colsSelected=1:86
    )
    return approach
end
