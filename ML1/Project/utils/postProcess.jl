
"""Retrieves the metrics for a specific Evaluation (result of architecture/problem run).
And store them in a csv file for further analysis."""
function storeResultData(approachName::String, result::Vector{Evaluation}; folder::String="./result/")
    appName = String[]
    modelName = String[]
    evaluationAccuracyScore = Float64[]
    trainedAccuracy = Float64[]
    validationAccuracy = Float64[]    
  
    ############## confusion matrix #############
    evaluationConfMatrixAccuracy = Float64[]
    evaluationConfMatrixErrorRate = Float64[]
    evaluationConfMatrixSensitivity = Float64[]
    evaluationConfMatrixSpecificity = Float64[]
    evaluationConfMatrixPpv = Float64[]
    evaluationConfMatrixNpv = Float64[]
    evaluationConfMatrixFScore = Float64[]
    
    ############ scikitTraining ###############
    scikitTrainingAccuracy= Float64[]
    scikitTrainingAccuracystd= Float64[]
    scikitTrainingErrorrate= Float64[]
    scikitTrainingFscore= Float64[]
    scikitTrainingFscorestd= Float64[]
    scikitTrainingNegativepredictivevalue= Float64[]
    scikitTrainingPositivepredictivevalue= Float64[]
    scikitTrainingPositivepredictivevaluestd= Float64[]
    scikitTrainingSensitivity= Float64[]
    scikitTrainingSensitivitystd= Float64[]
    scikitTrainingSpecificity= Float64[]
    
    ############ scikitValidation ###############
    scikitValidationAccuracy= Float64[]
    scikitValidationAccuracystd= Float64[]
    scikitValidationErrorrate= Float64[]
    scikitValidationFscore= Float64[]
    scikitValidationFscorestd= Float64[]
    scikitValidationNegativepredictivevalue= Float64[]
    scikitValidationPositivepredictivevalue= Float64[]
    scikitValidationPositivepredictivevaluestd= Float64[]
    scikitValidationSensitivity= Float64[]
    scikitValidationSensitivitystd= Float64[]
    scikitValidationSpecificity= Float64[]
    
    ############ scikitEvaluation ###############
    scikitEvaluationAccuracy= Float64[]
    scikitEvaluationAccuracystd= Float64[]
    scikitEvaluationErrorrate= Float64[]
    scikitEvaluationFscore= Float64[]
    scikitEvaluationFscorestd= Float64[]
    scikitEvaluationNegativepredictivevalue= Float64[]
    scikitEvaluationPositivepredictivevalue= Float64[]
    scikitEvaluationPositivepredictivevaluestd= Float64[]
    scikitEvaluationSensitivity= Float64[]
    scikitEvaluationSensitivitystd= Float64[]
    scikitEvaluationSpecificity= Float64[]
    
    for res in result
        push!(appName, approachName)
        push!(modelName,res.model.name)
        push!(trainedAccuracy,res.model.trainingScore)  
        push!(validationAccuracy,res.model.validationScore) 
        
        # conf matrix
        push!(evaluationConfMatrixAccuracy,res.confMatrix.accuracy) 
        push!(evaluationConfMatrixErrorRate,res.confMatrix.errorRate)  
        push!(evaluationConfMatrixSensitivity,res.confMatrix.sensitivity)  
        push!(evaluationConfMatrixSpecificity,res.confMatrix.specificity)  
        push!(evaluationConfMatrixPpv,res.confMatrix.ppv)  
        push!(evaluationConfMatrixNpv,res.confMatrix.npv)  
        push!(evaluationConfMatrixFScore,res.confMatrix.fScore) 
        
        ############ scikitTraining ###############
        push!(scikitTrainingAccuracy,res.model.trainingMetrics.accuracy)
        push!(scikitTrainingAccuracystd,res.model.trainingMetrics.accuracySTD)
        push!(scikitTrainingErrorrate,res.model.trainingMetrics.errorRate)
        push!(scikitTrainingFscore,res.model.trainingMetrics.fScore)
        push!(scikitTrainingFscorestd,res.model.trainingMetrics.fScoreSTD)
        push!(scikitTrainingNegativepredictivevalue,res.model.trainingMetrics.negativePredictiveValue)
        push!(scikitTrainingPositivepredictivevalue,res.model.trainingMetrics.positivePredictiveValue)
        push!(scikitTrainingPositivepredictivevaluestd,res.model.trainingMetrics.positivePredictiveValueSTD)
        push!(scikitTrainingSensitivity,res.model.trainingMetrics.sensitivity)
        push!(scikitTrainingSensitivitystd,res.model.trainingMetrics.sensitivitySTD)
        push!(scikitTrainingSpecificity,res.model.trainingMetrics.specificity)
        
        ############ scikitValidation ###############
        push!(scikitValidationAccuracy,res.model.validationMetrics.accuracy)
        push!(scikitValidationAccuracystd,res.model.validationMetrics.accuracySTD)
        push!(scikitValidationErrorrate,res.model.validationMetrics.errorRate)
        push!(scikitValidationFscore,res.model.validationMetrics.fScore)
        push!(scikitValidationFscorestd,res.model.validationMetrics.fScoreSTD)
        push!(scikitValidationNegativepredictivevalue,res.model.validationMetrics.negativePredictiveValue)
        push!(scikitValidationPositivepredictivevalue,res.model.validationMetrics.positivePredictiveValue)
        push!(scikitValidationPositivepredictivevaluestd,res.model.validationMetrics.positivePredictiveValueSTD)
        push!(scikitValidationSensitivity,res.model.validationMetrics.sensitivity)
        push!(scikitValidationSensitivitystd,res.model.validationMetrics.sensitivitySTD)
        push!(scikitValidationSpecificity,res.model.validationMetrics.specificity)
        
        ############ scikitEvaluation ###############        
        push!(scikitEvaluationAccuracy,res.evaluateMetrics.accuracy)
        push!(scikitEvaluationAccuracystd,res.evaluateMetrics.accuracySTD)
        push!(scikitEvaluationErrorrate,res.evaluateMetrics.errorRate)
        push!(scikitEvaluationFscore,res.evaluateMetrics.fScore)
        push!(scikitEvaluationFscorestd,res.evaluateMetrics.fScoreSTD)
        push!(scikitEvaluationNegativepredictivevalue,res.evaluateMetrics.negativePredictiveValue)
        push!(scikitEvaluationPositivepredictivevalue,res.evaluateMetrics.positivePredictiveValue)
        push!(scikitEvaluationPositivepredictivevaluestd,res.evaluateMetrics.positivePredictiveValueSTD)
        push!(scikitEvaluationSensitivity,res.evaluateMetrics.sensitivity)
        push!(scikitEvaluationSensitivitystd,res.evaluateMetrics.sensitivitySTD)
        push!(scikitEvaluationSpecificity,res.evaluateMetrics.specificity)

    end

    df = DataFrame()
    df.approach=appName
    df.modelName=modelName
    df.trainedAccuracy=trainedAccuracy
    df.validationAccuracy=validationAccuracy
    df.evaluationConfMatrixAccuracy=evaluationConfMatrixAccuracy
    df.evaluationConfMatrixErrorRate=evaluationConfMatrixErrorRate
    df.evaluationConfMatrixSensitivity=evaluationConfMatrixSensitivity
    df.evaluationConfMatrixSpecificity=evaluationConfMatrixSpecificity
    df.evaluationConfMatrixPpv=evaluationConfMatrixPpv
    df.evaluationConfMatrixNpv=evaluationConfMatrixNpv
    df.evaluationConfMatrixFScore=evaluationConfMatrixFScore    
    
    ############ scikitTraining ###############
    df.scikitTrainingAccuracy=scikitTrainingAccuracy
    df.scikitTrainingAccuracystd=scikitTrainingAccuracystd
    df.scikitTrainingErrorrate=scikitTrainingErrorrate
    df.scikitTrainingFscore=scikitTrainingFscore
    df.scikitTrainingFscorestd=scikitTrainingFscorestd
    df.scikitTrainingNegativepredictivevalue=scikitTrainingNegativepredictivevalue
    df.scikitTrainingPositivepredictivevalue=scikitTrainingPositivepredictivevalue
    df.scikitTrainingPositivepredictivevaluestd=scikitTrainingPositivepredictivevaluestd
    df.scikitTrainingSensitivity=scikitTrainingSensitivity
    df.scikitTrainingSensitivitystd=scikitTrainingSensitivitystd
    df.scikitTrainingSpecificity=scikitTrainingSpecificity
    
    ############ scikitValidation ###############
    df.scikitValidationAccuracy=scikitValidationAccuracy
    df.scikitValidationAccuracystd=scikitValidationAccuracystd
    df.scikitValidationErrorrate=scikitValidationErrorrate
    df.scikitValidationFscore=scikitValidationFscore
    df.scikitValidationFscorestd=scikitValidationFscorestd
    df.scikitValidationNegativepredictivevalue=scikitValidationNegativepredictivevalue
    df.scikitValidationPositivepredictivevalue=scikitValidationPositivepredictivevalue
    df.scikitValidationPositivepredictivevaluestd=scikitValidationPositivepredictivevaluestd
    df.scikitValidationSensitivity=scikitValidationSensitivity
    df.scikitValidationSensitivitystd=scikitValidationSensitivitystd
    df.scikitValidationSpecificity=scikitValidationSpecificity

    
    ############ scikitEvaluation ###############
    df.scikitEvaluationAccuracy=scikitEvaluationAccuracy
    df.scikitEvaluationAccuracystd=scikitEvaluationAccuracystd
    df.scikitEvaluationErrorrate=scikitEvaluationErrorrate
    df.scikitEvaluationFscore=scikitEvaluationFscore
    df.scikitEvaluationFscorestd=scikitEvaluationFscorestd
    df.scikitEvaluationNegativepredictivevalue=scikitEvaluationNegativepredictivevalue
    df.scikitEvaluationPositivepredictivevalue=scikitEvaluationPositivepredictivevalue
    df.scikitEvaluationPositivepredictivevaluestd=scikitEvaluationPositivepredictivevaluestd
    df.scikitEvaluationSensitivity=scikitEvaluationSensitivity
    df.scikitEvaluationSensitivitystd=scikitEvaluationSensitivitystd
    df.scikitEvaluationSpecificity=scikitEvaluationSpecificity

    dir = folder
    file = string(dir, approachName, ".csv")
    isdir(dir) || mkdir(dir)
    CSV.write(file, df, header=true)

end
