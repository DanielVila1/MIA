

"""Reads the CSV train and test datasets from the given file paths."""
function readDatasets(trainPath, testPath)
    trainData = CSV.read(trainPath, DataFrame, header=false, delim=",")
    testData = CSV.read(testPath, DataFrame, header=false, delim=",")
    return (train=trainData, test=testData)
end

"""Pre-process the train dataset to keep a percentage of the samples randomly chosen and
create copies of the samples of each class."""
function preprocessTrainDataset(trainSet; class_column=188, multiply_class_samples=[])
    if multiply_class_samples == []
        return trainSet
    end
    num_samples = size(trainSet)[1]
    grouped = groupby(trainSet, class_column)
    num_classes = size(grouped)[1]
    multipliers_length = length(multiply_class_samples)
    if multipliers_length < num_classes
        for i=multipliers_length:num_classes
            append!(multiply_class_samples, 1.0)
        end
    end
    result = similar(trainSet, 0)
    original_sizes = []
    final_sizes = []
    for i=1:size(grouped)[1]
        group = grouped[i]
        push!(original_sizes, size(group)[1])
        group_num_samples = nrow(group)
        multiply_class_i = multiply_class_samples[i]
        if multiply_class_i < 1
            group_num_samples_keep = trunc(Int, group_num_samples * multiply_class_i)
            indexes = shuffle(1:group_num_samples)[1:group_num_samples_keep]
            sampled_group = group[indexes, :]
            push!(final_sizes, group_num_samples_keep)
            result = vcat(result, sampled_group)
        else
            times = trunc(Int, multiply_class_i)
            for j=1:times
                result = vcat(result, group)
            end
            push!(final_sizes, group_num_samples * times)
        end
    end
    return result, original_sizes, final_sizes
end


# Sample code to test the readDatasets and preprocessTrainDataset functions.

if false
    df = DataFrame(column_1 = ["value_1", "value_2", "value_3", "value_4", "value_4"], 
                column_2 = ["value_1", "value_2", "value_3", "value_4", "value_4"],
                column_3 = ["class_1", "class_2", "class_3", "class_3", "class_3"],)
    pdf, _, _ = preprocessTrainDataset(df, class_column=3, multiply_class_samples=[2,1,1])
    println(size(pdf))
    show(pdf)
    t = readDatasets("../dataset/mitbih_train.csv", "../dataset/mitbih_test.csv")
    println(size(t.train))
    println(size(t.test))
    tp, os, fs = preprocessTrainDataset(t.train, multiply_class_samples=[0.5,2,1,5,1])
    println(size(tp))
    println(os)
    println(fs)
end
