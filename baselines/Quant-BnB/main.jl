using LinearAlgebra
using Printf
using Random
using DataFrames
using CSV
using HTTP
using Statistics
using ScikitLearn
using ScikitLearn.CrossValidation: train_test_split

try
    using StatsBase
    global has_statsbase = true
catch
    global has_statsbase = false
end

include("QuantBnB-2D.jl")
include("QuantBnB-3D.jl")
include("gen_data.jl")
include("lowerbound_middle.jl")
include("Algorithms.jl")

# Minimal helper functions
function load_dataset(url, target_column)
    data = CSV.File(HTTP.get(url).body, header=1) |> DataFrame

    # Infer the target column as the last one if -1 is given
    if target_column == -1
        target_column = size(data, 2)
    end

    # Convert features to Float64 or integer for categorical
    for col in 1:(size(data, 2)-1)
        if eltype(data[!, col]) <: AbstractString
            try
                data[!, col] = parse.(Float64, data[!, col])
            catch
                categories = unique(data[!, col])
                data[!, col] = [findfirst(==(x), categories) for x in data[!, col]]
            end
        else
            data[!, col] = float.(data[!, col])
        end
    end

    # Handle target column (convert strings to integer classes)
    if eltype(data[!, target_column]) <: AbstractString
        categories = unique(data[!, target_column])
        data[!, target_column] = [findfirst(==(x), categories) for x in data[!, target_column]]
    else
        data[!, target_column] = Int.(data[!, target_column])
    end

    data = dropmissing(data)  # remove rows with missing values
    X = Matrix(data[:, 1:(end-1)])
    y = data[:, end]
    return X, Float64.(y)
end

function count_nodes(tree)
    if typeof(tree) <: Number || (isa(tree, Array) && eltype(tree) <: Number)
        return 1  # Leaf node
    elseif length(tree) == 4
        return 1 + count_nodes(tree[3]) + count_nodes(tree[4])
    else
        println("Warning: Unexpected tree structure encountered")
        return 1
    end
end

function print_tree(tree, indent="")
    if length(tree) == 4 && !(tree[3] isa Number || tree[3] isa Vector)
        println(indent, "Feature ", tree[1], " < ", tree[2])
        print_tree(tree[3], indent * "  ")
        print_tree(tree[4], indent * "  ")
    else
        println(indent, "Leaf: ", tree)
    end
end

# Datasets and seeds
datasets = [
    ("Iris", "https://archive.ics.uci.edu/static/public/53/data.csv", -1),
    ("Breast Cancer Wisconsin", "https://archive.ics.uci.edu/static/public/17/data.csv", -1),
    ("Wine", "https://archive.ics.uci.edu/static/public/109/data.csv", -1),
    ("Raisin", "https://archive.ics.uci.edu/static/public/850/data.csv", -1)
]
seeds = [1, 2, 3, 4, 5]

# -- Depth-2 trees --
println("Test Quant-BnB on depth-2 trees for classification problems")

results_2d = Dict()

for (name, url, target_column) in datasets
    println("\nDataset: $name")
    X, y = load_dataset(url, target_column)

    cart_accuracies = Float64[]
    qbnb_accuracies = Float64[]
    cart_sizes = Int[]
    qbnb_sizes = Int[]
    times_list = Float64[]

    for seed in seeds
        Random.seed!(seed)

        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

        n_classes = length(unique(y))
        Y_train_onehot = float.(hcat([Y_train .== c for c in 1:n_classes]...))
        Y_test_onehot = float.(hcat([Y_test .== c for c in 1:n_classes]...))

        n_train, m = size(Y_train_onehot)
        n_test, _ = size(Y_test_onehot)

        start = time()
        gre_train, gre_tree = greedy_tree(X_train, Y_train_onehot, 2, "C")
        opt_train, opt_tree = QuantBnB_2D(X_train, Y_train_onehot, 3, gre_train*(1+1e-6), 2, 0.2, nothing, "C", false)
        push!(times_list, time() - start)

        gre_test = sum(argmax(Y_test_onehot, dims=2) .!= argmax(tree_eval(gre_tree, X_test, 2, m), dims=2))
        opt_test = sum(argmax(Y_test_onehot, dims=2) .!= argmax(tree_eval(opt_tree, X_test, 2, m), dims=2))

        cart_acc = 1 - gre_test/n_test
        qbnb_acc = 1 - opt_test/n_test

        push!(cart_accuracies, cart_acc)
        push!(qbnb_accuracies, qbnb_acc)
        push!(cart_sizes, count_nodes(gre_tree))
        push!(qbnb_sizes, count_nodes(opt_tree))

        @printf("Seed: %d, CART train/test acc: %.3f / %.3f, Quant-BnB train/test acc: %.3f / %.3f\n",
                seed, 1-gre_train/n_train, cart_acc, 1-opt_train/n_train, qbnb_acc)
        println("CART tree structure:")
        print_tree(gre_tree)
        println("Quant-BnB tree structure:")
        print_tree(opt_tree)
        println("---")
    end

    results_2d[name] = Dict(
        "cart" => Dict(
            "mean_accuracy" => mean(cart_accuracies),
            "std_accuracy" => std(cart_accuracies),
            "mean_model_size" => mean(cart_sizes),
            "std_model_size" => std(cart_sizes)
        ),
        "qbnb" => Dict(
            "mean_accuracy" => mean(qbnb_accuracies),
            "std_accuracy" => std(qbnb_accuracies),
            "mean_model_size" => mean(qbnb_sizes),
            "std_model_size" => std(qbnb_sizes),
            "mean_time" => mean(times_list),
            "std_time" => std(times_list)
        )
    )

    println("\nResults for $name:")
    for method in ["CART", "Quant-BnB"]
        key = lowercase(method) == "cart" ? "cart" : "qbnb"
        @printf("%s - Mean Accuracy: %.4f, Std: %.4f\n",
                method, results_2d[name][key]["mean_accuracy"], results_2d[name][key]["std_accuracy"])
        @printf("%s - Mean Model Size: %.4f, Std: %.4f\n",
                method, results_2d[name][key]["mean_model_size"], results_2d[name][key]["std_model_size"])
        if key == "qbnb"
            @printf("%s - Mean Time: %.4f, Std: %.4f\n",
                    method, results_2d[name][key]["mean_time"], results_2d[name][key]["std_time"])
        end
    end
end

# -- Depth-3 trees --
println("\nTest Quant-BnB on depth-3 trees for classification problems")

results_3d = Dict()

for (name, url, target_column) in datasets
    println("\nDataset: $name")
    X, y = load_dataset(url, target_column)

    cart_accuracies = Float64[]
    qbnb_accuracies = Float64[]
    cart_sizes = Int[]
    qbnb_sizes = Int[]

    for seed in seeds
        Random.seed!(seed)

        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        n_classes = length(unique(y))
        Y_train_onehot = float.(hcat([Y_train .== c for c in 1:n_classes]...))
        Y_test_onehot = float.(hcat([Y_test .== c for c in 1:n_classes]...))

        n_train, m = size(Y_train_onehot)
        n_test, _ = size(Y_test_onehot)

        gre_train, gre_tree = greedy_tree(X_train, Y_train_onehot, 3, "C")
        opt_train, opt_tree = QuantBnB_3D(X_train, Y_train_onehot, 3, 3, gre_train*(1+1e-6), 0, 0, nothing, "C", 300)

        gre_test = sum(argmax(Y_test_onehot, dims=2) .!= argmax(tree_eval(gre_tree, X_test, 3, m), dims=2))
        opt_test = sum(argmax(Y_test_onehot, dims=2) .!= argmax(tree_eval(opt_tree, X_test, 3, m), dims=2))

        cart_acc = 1 - gre_test/n_test
        qbnb_acc = 1 - opt_test/n_test

        push!(cart_accuracies, cart_acc)
        push!(qbnb_accuracies, qbnb_acc)
        push!(cart_sizes, count_nodes(gre_tree))
        push!(qbnb_sizes, count_nodes(opt_tree))

        @printf("Seed: %d, CART train/test acc: %.3f / %.3f, Quant-BnB train/test acc: %.3f / %.3f\n",
                seed, 1-gre_train/n_train, cart_acc, 1-opt_train/n_train, qbnb_acc)
        println("CART tree structure:")
        print_tree(gre_tree)
        println("Quant-BnB tree structure:")
        print_tree(opt_tree)
        println("---")
    end

    results_3d[name] = Dict(
        "cart" => Dict(
            "mean_accuracy" => mean(cart_accuracies),
            "std_accuracy" => std(cart_accuracies),
            "mean_model_size" => mean(cart_sizes),
            "std_model_size" => std(cart_sizes)
        ),
        "qbnb" => Dict(
            "mean_accuracy" => mean(qbnb_accuracies),
            "std_accuracy" => std(qbnb_accuracies),
            "mean_model_size" => mean(qbnb_sizes),
            "std_model_size" => std(qbnb_sizes)
        )
    )

    println("\nResults for $name:")
    for method in ["CART", "Quant-BnB"]
        key = lowercase(method) == "cart" ? "cart" : "qbnb"
        @printf("%s - Mean Accuracy: %.4f, Std: %.4f\n",
                method, results_3d[name][key]["mean_accuracy"], results_3d[name][key]["std_accuracy"])
        @printf("%s - Mean Model Size: %.4f, Std: %.4f\n",
                method, results_3d[name][key]["mean_model_size"], results_3d[name][key]["std_model_size"])
    end
end

# Example: test depth-3 trees on a local classification dataset
X_train, X_test, Y_train, Y_test = generate_realdata(string("./dataset/class/", "bidding", ".json"))
n_train, m = size(Y_train)
n_test, _ = size(Y_test)

gre_train, gre_tree = greedy_tree(X_train, Y_train, 3, "C")
opt_train, opt_tree = QuantBnB_3D(X_train, Y_train, 3, 3, gre_train*(1+1e-6), 0, 0, nothing, "C", 300)

gre_test = sum((Y_test .- tree_eval(gre_tree, X_test, 3, m)) .> 0)
opt_test = sum((Y_test .- tree_eval(opt_tree, X_test, 3, m)) .> 0)

@printf("\nLocal dataset 'bidding', CART train/test acc: %.3f / %.3f, Quant-BnB train/test acc: %.3f / %.3f\n",
        1-gre_train/n_train, 1-gre_test/n_test, 1-opt_train/n_train, 1-opt_test/n_test)
