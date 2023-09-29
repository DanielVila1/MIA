
include("../utils/approach.jl")
include("../utils/approachNew.jl")


# Functions to create the models for the approach.

function getAnns3()::Array{ANN}
    return [
        buildANN(layers=[4,2]),
        buildANN(layers=[4,4]),
        buildANN(layers=[6,6]),
        buildANN(layers=[8,8]),
        buildANN(layers=[16,16]),
        buildANN(layers=[8]),
        buildANN(layers=[12]),
        buildANN(layers=[32]),
    ]
end

function getSvms3()::Array{SVM}
    return [
        buildSVM(kernel="sigmoid", C=1.5),
        buildSVM(kernel="sigmoid", C=2.0),
        buildSVM(kernel="sigmoid", C=3.5),
        buildSVM(kernel="sigmoid", C=5.0),
        buildSVM(kernel="rbf", C=15.0),
        buildSVM(kernel="rbf", C=20.0),
        buildSVM(kernel="rbf", C=50.0),
        buildSVM(kernel="rbf", C=100.0),
    ]
end



function getDts3()::Array{DT}
    return [
        buildDT(depth=3),
        buildDT(depth=4),
        buildDT(depth=5),
        buildDT(depth=6),
        buildDT(depth=7),
        buildDT(depth=8),
    ]
end

function getkNNs3()::Array{KNN}
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

function createApproach3()
    approach = createApproach(
        name="approach3",
        multiplyClassSamples=[0.5,2,1,5,1], 
        anns=getAnns3(),
        svms=getSvms3(),
        dts=getDts3(),
        knns=getkNNs3(), 
        useKFoldCrossValidation=true,
        kFolds=5,
        ensembles=[HardVoting, SoftVoting, RandomForest],
        reduceDimensionality=false
    )
    return approach
end
