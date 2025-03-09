using LinearAlgebra
using JSON

function tree_eval(tree, X, D, m)
    n, p = size(X)
    y0 = zeros(n, m)
    
    if typeof(tree) <: Number  # If tree is a single number (leaf node for regression)
        return fill(tree, n, m)
    elseif length(tree) != 4 || (typeof(tree[3]) <: Number && typeof(tree[4]) <: Number)
        # Leaf node for classification
        return repeat(reshape(tree, 1, :), n, 1)
    end
    
    f = tree[1]
    b = tree[2]
    idx1, idx2 = treesplit(x -> x<b, X[:,f])
    y0[idx1,:] = tree_eval(tree[3], X[idx1,:], D-1, m)
    y0[idx2,:] = tree_eval(tree[4], X[idx2,:], D-1, m)
    return y0    
end

function generate_realdata(name)
    realdata = JSON.parsefile(name)
    Ntrain = size(realdata["Xtrain"])[1]
    Ntest = size(realdata["Xtest"])[1]
    X_train = zeros(Ntrain, realdata["F"])
    X_test = zeros(Ntest, realdata["F"])
    Y_train = zeros(Ntrain, realdata["C"])
    Y_test = zeros(Ntest, realdata["C"])
    for i=1:Ntrain
        X_train[i,:] = realdata["Xtrain"][i]
    end
    for i=1:Ntest
        X_test[i,:] = realdata["Xtest"][i]
    end
    if realdata["treetype"] == "R" 
        for i=1:Ntrain
            Y_train[i,:] = realdata["Ytrain"][i]
        end
        for i=1:Ntest
            Y_test[i,:] = realdata["Ytest"][i]
        end
    else
        for i=1:Ntrain
            Y_train[i, Int(realdata["Ytrain"][i]) + 1] = 1
        end
        for i=1:Ntest
            Y_test[i, Int(realdata["Ytest"][i]) + 1] = 1
        end
    end

    n, p = size(X_train)

    goodfeature = Vector{Int64}()
    for i = 1:p
        if length(unique(X_train[:,i])) >= 2
            append!(goodfeature, i)
        end
    end
    X_train = X_train[:,goodfeature]
    X_test = X_test[:,goodfeature]


    return X_train, X_test, Y_train, Y_test
end



