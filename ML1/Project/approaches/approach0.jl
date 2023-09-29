
include("../utils/approach.jl")


function createApproach0()
    approach = createApproach(
        name="approach0",
        #multiplyClassSamples=[0.1,0.4,0.2,1,0.2], 
        multiplyClassSamples=[1,10,10,10,10], 
        anns=[],#buildANN(layers=[5])], 
        svms=[],#buildSVM(kernel="rbf", C=1.0, linear=true)], 
        dts=[buildDT(depth=5)], 
        knns=[],#buildKNN(neighbors=3)],
        ensembles=[],#HardVoting],
        useKFoldCrossValidation=false,
        reduceDimensionality=false,
        validationRatio=0.3,
        showPlots=true,
        customs=[]#buildCustom(binary=buildDT(depth=5), multi=buildDT(depth=5))]
    )
    return approach
end
