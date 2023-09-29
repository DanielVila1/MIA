
"""Returns a named tuple with all the relevant metrics for the given evaluation."""
function metricsToRowTuple(approachName::String, evaluation::Evaluation)
    return (
        approach=approachName,
        model=evaluation.model.name,
        trainAcc=evaluation.model.trainingMetrics.accuracy,
        valAcc=evaluation.model.validationMetrics.accuracy,
        testAcc=evaluation.evaluateMetrics.accuracy,
        trainFS=evaluation.model.trainingMetrics.fScore,
        valFS=evaluation.model.validationMetrics.fScore,
        testFS=evaluation.evaluateMetrics.fScore,
        trainPPV=evaluation.model.trainingMetrics.positivePredictiveValue,
        valPPV=evaluation.model.validationMetrics.positivePredictiveValue,
        testPPV=evaluation.evaluateMetrics.positivePredictiveValue,
        trainRec=evaluation.model.trainingMetrics.sensitivity,
        valRec=evaluation.model.validationMetrics.sensitivity,
        testRec=evaluation.evaluateMetrics.sensitivity
    )
end

"""Stores the provided evaluation results in a CSV file."""
function storeResultDataNew(approachName::String, result::Vector{Evaluation}; folder::String="./result/")
    rows = map(x -> metricsToRowTuple(approachName, x), result)
    df = DataFrame(rows)
    dir = folder
    file = string(dir, approachName, ".csv")
    isdir(dir) || mkdir(dir)
    t = (c,v)->(isa(v,Float64) ? @sprintf("%1.3f", v) : v)
    CSV.write(file, df, header=true, transform=t)
end
