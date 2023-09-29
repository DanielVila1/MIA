
include("../utils/approach.jl")


# Functions to create the models for the approach.

function getSvms5b()::Array{SVM}
    return [
        buildSVM(C=1.5, linear=true),
        buildSVM(C=2.0, linear=true),
        buildSVM(kernel="sigmoid", C=1.5, kernelGamma=2, maxIter=50, bagged=true),
        buildSVM(kernel="sigmoid", C=2.0, kernelGamma=3, maxIter=50, bagged=true),
        buildSVM(kernel="rbf", C=1.5, maxIter=50, bagged=true),
        buildSVM(kernel="rbf", C=2.0, maxIter=50, bagged=true),
        buildSVM(kernel="rbf", C=1.5, kernelGamma=2, maxIter=50, bagged=true),
        buildSVM(kernel="rbf", C=2.0, kernelGamma=3, maxIter=50, bagged=true),
    ]
end


# Function to create the approach.

function createApproach5b()
    approach = createApproach(
        name="approach5b",
        multiplyClassSamples=[0.5,5,5,5,5], 
        anns=[],
        svms=getSvms5b(),
        dts=[],
        knns=[], 
        useKFoldCrossValidation=false,
        validationRatio=0.2,
        ensembles=[],
        reduceDimensionality=true,
        dimReduction=1.87,
        dimensionalityReduction=DR_PCA,
        customs=[]
    )
    return approach
end
