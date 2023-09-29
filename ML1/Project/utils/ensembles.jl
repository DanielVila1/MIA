
"""Creates a HARD Voting ensemble model."""
function createHardVotingEnsemble(models, weights)
    ensemble = VotingClassifier(estimators=models, n_jobs=1, voting="hard",verbose=true)
    return ensemble
end

"""Creates a SOFT Voting ensemble model."""
function createSoftVotingEnsemble(models, weights)
    ensemble = VotingClassifier(estimators=models, n_jobs=1, voting="soft", weights=weights,verbose=true)
    return ensemble
end

"""Creates a Random Forest ensemble model."""
function createRandomForest()
    ensemble = RandomForestClassifier(n_estimators=100, max_depth=5)
    return ensemble
end

"""Creates a Gradient Boosting ensemble model."""
function createGradientBoosting()
    ensemble = GradientBoostingClassifier(n_estimators=10)
    return ensemble
end

"""Creates an XGBoost ensemble model."""
function createXGBoost()
    ensemble = xgb.XGBClassifier(n_estimators=100, max_depth=10, learning_rate=1, objective="binary:logistic")
    return ensemble
end