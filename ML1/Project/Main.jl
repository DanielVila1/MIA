
include("utils/approach.jl")
include("utils/approachNew.jl")
include("approaches/approach0.jl")
include("approaches/approach1.jl")
include("approaches/approach2.jl")
include("approaches/approach2b.jl")
include("approaches/approach3.jl")
include("approaches/approach4.jl")
include("approaches/approach5.jl")
include("approaches/approach5b.jl")


# Functions to create each approach and run it for the given problem.

function runApproach0(problem::Problem)
    approach = createApproach0()
    runAllPerModel(problem, approach)
end

function runApproach1(problem::Problem)
    approach = createApproach1()
    runAllPerModel(problem, approach)
end

function runApproach2(problem::Problem)
    approach = createApproach2()
    runAllPerModel(problem, approach)
end

function runApproach2b(problem::Problem)
    approach = createApproach2b()
    runAllPerModel(problem, approach)
end

function runApproach3(problem::Problem)
    approach3 = createApproach3()
    runAllPerModel(problem, approach3)
end

function runApproach4(problem::Problem)
    approach = createApproach4()
    runAllPerModel(problem, approach)
end

function runApproach5(problem::Problem)
    approach = createApproach5()
    runAllPerModel(problem, approach)
end

function runApproach5b(problem::Problem)
    approach = createApproach5b()
    runAllPerModel(problem, approach)
end


# Main function.

"""Runs all the approaches for all the classification problems."""
function runApproaches()
    multiClassProblem = buildProblem(
        trainSetPath="./dataset/mitbih_train.csv",
        testSetPath="./dataset/mitbih_test.csv"
    )
    binaryProblem = buildProblem(
        trainSetPath="./dataset/mitbih_train.csv",
        testSetPath="./dataset/mitbih_test.csv",
        isBinaryClassification=true
    )
    # Test approach
    runApproach0(multiClassProblem)
    #runApproach0(binaryProblem)
    # Approaches with binary classification problem 
    runApproach1(binaryProblem)
    # Approaches with multi-class classification problem 
    runApproach2(multiClassProblem)
    runApproach2b(multiClassProblem)
    runApproach3(multiClassProblem)
    runApproach4(multiClassProblem)
    runApproach5(multiClassProblem)
    runApproach5b(multiClassProblem)
end


runApproaches()
