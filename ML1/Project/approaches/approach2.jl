include("../utils/approach.jl")
include("../utils/approachNew.jl")

function getAnns2()::Array{ANN}
    return [
        buildANN(layers=[4,4]),
        buildANN(layers=[2,2]),
        buildANN(layers=[16,8]),
        buildANN(layers=[4]),
        buildANN(layers=[8]),
        buildANN(layers=[16]),
        buildANN(layers=[32]),
        buildANN(layers=[64]),
    ]
end

function getSvms2()::Array{SVM}
    return [
        buildSVM(kernel="sigmoid", C=0.5, linear=true, probability=true),
        buildSVM(kernel="sigmoid", C=1.0, linear=true, probability=true),
        buildSVM(kernel="sigmoid", C=1.5, linear=true, probability=true),
        buildSVM(kernel="sigmoid", C=2.0, linear=true, probability=true),
        buildSVM(kernel="sigmoid", C=2.5, linear=true, probability=true),
        buildSVM(kernel="sigmoid", C=3.0, linear=true, probability=true),
        buildSVM(kernel="sigmoid", C=3.5, linear=true, probability=true),
        buildSVM(kernel="sigmoid", C=4.0, linear=true, probability=true),
    ]
end

function getDts2()::Array{DT}
    return [
        buildDT(depth=3),
        buildDT(depth=4),
        buildDT(depth=5),
        buildDT(depth=6),
        buildDT(depth=7),
        buildDT(depth=8),
    ]
end

function getkNNs2()::Array{KNN}
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

function createApproach2()
    approach = createApproach(
        name="approach2",
        multiplyClassSamples=[0.3,2.5,1.5,5,1.5], 
        anns=getAnns2(),
        svms=getSvms2(),
        dts=getDts2(),
        knns=getkNNs2(), 
        useKFoldCrossValidation=false, 
        ensembles=[XGBoost, HardVoting, SoftVoting],
        reduceDimensionality=false,
        validationRatio=0.3,
    )
    return approach
end
