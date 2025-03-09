using LinearAlgebra
include("gen_data.jl")

## Sorting 
function presort(X)
    n, p = size(X)
    X_sort = zeros(n, p)
    X_sortperm = zeros(Int, n, p)
    X_invperm = zeros(Int, n, p)
    X_count = zeros(Int, n, p)
    X_invcount = Array{Any,1}(undef, p)
    X_permcount = zeros(Int, n, p)
    X_dict1 = Array{Any,1}(undef, p)
    X_dict2 = Array{Any,1}(undef, p)
    for i = 1:p
        X_sort[:, i] = sort(X[:,i])
        od = sortperm(X[:,i])
        X_sortperm[:,i] = od
        X_invperm[od,i] = 1:n
        X_invcount[i] = Vector{Int64}() 
        count = 1
        append!(X_invcount[i], 0)
        if n >= 1
            X_count[1,i] = count
            for j=2:n
                if X_sort[j-1,i] < X_sort[j,i]
                    count += 1
                    append!(X_invcount[i], j-1)
                end
                X_count[j,i] = count
            end
            append!(X_invcount[i], n)
        end
        X_dict1[i] = Dict(zip(X_sort[X_invcount[i][2:end],i],X_invcount[i][1:end-1]))
        X_dict2[i] = Dict(zip(X_sort[X_invcount[i][2:end],i],X_invcount[i][2:end]))
        X_permcount[:,i] = X_count[X_invperm[:,i],i]
    end
    return X_sort, X_sortperm, X_invperm, X_count, X_invcount, X_dict1, X_dict2, X_permcount
end

## Get partitation threshold (X sorted)
function get_thres(X)
    n = length(X)
    f_count = zeros(Int,n)
    f_invcount = Vector{Int64}() 
    count = 1
    append!(f_invcount, 0)
    if n >= 1
        f_count[1] = count
        for j=2:n
            if X[j-1] < X[j]
                count += 1
                append!(f_invcount, j-1)
            end
            f_count[j] = count
        end
        append!(f_invcount, n)
    end
    return f_count, f_invcount
end

## One layer search
function one_pass_search_mr(X, y, idx, feature, type)
    
    ff = feature
    nn = length(idx)
    n, m = size(y)
    
    if nn == 0
        tmp = zeros(1,m)
        tmp[1] = 1
        return 0, 0, tmp, tmp
    end
    if nn == 1 
        return 0, 0, y[idx, :], y[idx, :]
    end
    if nn == 2
        if X[idx[1],ff] < X[idx[2],ff]
            return 0, (X[idx[1],ff] + X[idx[2],ff])/2, y[idx[1],:], y[idx[2],:]
        elseif X[idx[1],ff] > X[idx[2],ff]
            return 0, (X[idx[1],ff] + X[idx[2],ff])/2, y[idx[2],:], y[idx[1],:]
        else
            loss = sum((y[idx[1],:] .- y[idx[2],:]).^2) / 2.0
            if type == "R"
                mv = (y[idx[1],:] + y[idx[2],:]) ./ 2
            else
                mv = y[idx[1],:]
            end
            return loss, 0, mv, mv
        end
    end

    x_ordered, y_ordered = 0, 0
    if nn < n / 10
        X_sub = X[idx, ff]
        y_sub = y[idx, :]
        od = sortperm(X_sub)
        y_ordered = y_sub[od, :]
        x_ordered = X_sub[od]  
    else
        od = zeros(Int, nn+1)
        tmp = zeros(Int, n)
        tmp[X_invperm[idx,ff]] = idx
        count = 1
        @inbounds for i=1:n
            od[count] = tmp[i]
            count += (tmp[i] != zero(Int))
        end
        y_ordered = y[od[1:end-1], :]
        x_ordered = X[od[1:end-1],ff]
    end
    x_same = (x_ordered[1:end-1]-x_ordered[2:end]) .!= 0

    if type == "R"
        Cy = cumsum(y_ordered, dims = 1)
        RCy = Cy[nn, :]' .- Cy
        y2 = sum(y_ordered.^2)
        
        Cy_norms = reshape(sum(Cy.^2, dims=2), nn)[1:nn-1]
        RCy_norms = reshape(sum(RCy.^2, dims=2), nn)[1:nn-1]
        
        a = 1:nn-1
        second_terms = Cy_norms./a + RCy_norms./a[nn-1: -1:1]
        pos = argmax(second_terms .* x_same)
        if x_same[pos] != 0
            loss_best = y2 - second_terms[pos]
            mean_left_best = reshape(Cy[pos,:]./pos, (1,m))
            mean_right_best = reshape(RCy[pos,:]./(nn-pos), (1,m))
            threshold_best = (x_ordered[pos] + x_ordered[pos+1])/2
        else
            loss_best = y2 - sum(sum(y_ordered, dims = 1).^2) / nn
            mean_left_best = sum(y_ordered, dims = 1) / nn
            mean_right_best = mean_left_best
            threshold_best = 0
        end
    elseif type == "C"
        Cy = cumsum(y_ordered, dims = 1)
        RCy = Cy[nn, :]' .- Cy
        Cy_max = maximum(Cy,dims = 2)[1:nn-1]
        RCy_max = maximum(RCy,dims = 2)[1:nn-1]
        second_terms = Cy_max + RCy_max
        pos = argmax(second_terms .* x_same)

        if x_same[pos] != 0
            loss_best = nn .- second_terms[pos]
            mean_left_best  = zeros(1,m)
            mean_right_best  = zeros(1,m)
            mean_left_best[argmax(Cy[pos,:])] = 1
            mean_right_best[argmax(RCy[pos,:])] = 1
            threshold_best = (x_ordered[pos] + x_ordered[pos+1])/2

        else
            loss_best = nn - maximum(sum(y_ordered, dims=1))
            mean_left_best = zeros(1,m)
            mean_left_best[argmax(sum(y_ordered, dims=1))[2]] = 1
            mean_right_best = mean_left_best
            threshold_best = 0
        end
    elseif type == "G" # Information gain

        Cy = cumsum(y_ordered, dims = 1)
        RCy = Cy[nn, :]' .- Cy
        y_ent = -sum(Cy[nn,:] .* log2.(Cy[nn,:]./nn + 1e-8 .* (Cy[nn,:] .== 0)))
        
        Cy_ent = -reshape(sum(Cy .* log2.(Cy + 1e-8 .* (Cy .== 0)), dims=2), nn)[1:nn-1]
        RCy_ent = -reshape(sum(RCy .* log2.(RCy + 1e-8 .* (RCy .== 0)), dims=2), nn)[1:nn-1]
        a = Array(1:nn-1)
        n_ent = (a .* log2.(a) + (nn.-a) .* log2.(nn.-a))
        second_terms = y_ent .- (Cy_ent + RCy_ent + n_ent)
        pos = argmax(second_terms .* x_same)

        loss_best =  y_ent - second_terms[pos] 
        mean_left_best  = zeros(1,m)
        mean_right_best = zeros(1,m)
        mean_left_best[argmax(Cy[pos,:])] = 1
        mean_right_best[argmax(RCy[pos,:])] = 1
        threshold_best = (x_ordered[pos] + x_ordered[pos+1])/2

    else # missing value
        Ynan = y_ordered .!== NaN
        y_ordered .*= Ynan
        Nansum = cumsum(Ynan, dims = 1)[1:nn-1,:]
        NansumR = reverse(cumsum(Ynan[end:-1:1,:], dims = 1)[1:nn-1,:], dims = 1)
        Nansum += 1e-1 * (Nansum .== 0)
        NansumR += 1e-1 * (NansumR .== 0)
        
        Cy = cumsum(y_ordered, dims = 1)
        RCy = Cy[nn, :]' .- Cy
        y2 = sum(y_ordered.^2)
        
        Cy_norms = reshape(sum(Cy[1:nn-1,:].^2 ./Nansum, dims=2), nn-1)
        RCy_norms = reshape(sum(RCy[1:nn-1,:].^2 ./NansumR, dims=2), nn-1)
        second_terms = Cy_norms + RCy_norms
        pos = argmax(second_terms .* x_same)

        if x_same[pos] != 0
            loss_best = y2 - second_terms[pos]
            mean_left_best = reshape(Cy[pos,:]./Nansum[pos,:], (1,m))
            mean_right_best = reshape(RCy[pos,:]./NansumR[pos,:], (1,m))
            threshold_best = (x_ordered[pos] + x_ordered[pos+1])/2
        else
            loss_best = y2 - sum(sum(y_ordered, dims = 1) .^2 ./ sum(Ynan, dims = 1))
            mean_left_best = sum(y_ordered, dims = 1) ./ sum(Ynan, dims = 1)
            mean_right_best = mean_left_best
            threshold_best = 0
        end
    end
    
    return loss_best, threshold_best, mean_left_best, mean_right_best   
    
end

## Greedy tree -- fixed depth
function greedy_tree(X, y, D, type = "R")
    nn, p = size(X)
    _, m = size(y)
    if nn == 0
        best_tree = Array{Any,1}(undef, 4)
        best_tree[1] = 1
        best_tree[2] = 0
        if D == 1
            best_tree[3] = zeros(m)
            best_tree[4] = zeros(m)
            return 0, best_tree
        else
            _, best_left = greedy_tree(X, y, D-1, type)
            _, best_right = greedy_tree(X, y, D-1, type)
            best_tree[3] = best_left
            best_tree[4] = best_right
            return 0, best_tree
        end
    end
    if nn == 1
        best_tree = Array{Any,1}(undef, 4)
        best_tree[1] = 1
        best_tree[2] = 0
        if D == 1
            best_tree[3] = y[1,:]
            best_tree[4] = y[1,:]
            return 0, best_tree
        else
            _, best_left = greedy_tree(X, y, D-1, type)
            _, best_right = greedy_tree(X, y, D-1, type)
            best_tree[3] = best_left
            best_tree[4] = best_right
            return 0, best_tree
        end
    end
    
    loss_best = 1e30
    threshold_best = 0
    mean_left_best = zeros(m)
    mean_right_best = zeros(m)
    label_best = 1
    
    for ff = 1:p
    
        od = sortperm(X[:,ff])
        y_ordered = y[od, :]
        x_ordered = X[od,ff]  
   
        x_same = (x_ordered[1:end-1]-x_ordered[2:end]) .!= 0

        if type == "R"
            Cy = cumsum(y_ordered, dims = 1)
            RCy = Cy[nn, :]' .- Cy
            y2 = sum(y_ordered.^2)
            
            Cy_norms = reshape(sum(Cy.^2, dims=2), nn)[1:nn-1]
            RCy_norms = reshape(sum(RCy.^2, dims=2), nn)[1:nn-1]
            
            a = Array(1:nn-1)
            second_terms = Cy_norms./a + RCy_norms./a[nn-1: -1:1]
            pos = argmax(second_terms .* x_same)
            loss_cur = y2 - second_terms[pos]
            mean_left_cur = reshape(Cy[pos,:]./pos, (1,m))
            mean_right_cur = reshape(RCy[pos,:]./(nn-pos), (1,m))
            threshold_cur = (x_ordered[pos] + x_ordered[pos+1])/2
        elseif type == "C"
            Cy = cumsum(y_ordered, dims = 1)
            RCy = Cy[nn, :]' .- Cy
            Cy_norms = reshape(sum(Cy.^2, dims=2), nn)[1:nn-1]
            RCy_norms = reshape(sum(RCy.^2, dims=2), nn)[1:nn-1]

            a = Array(1:nn-1)
            second_terms = Cy_norms./a + RCy_norms./a[nn-1: -1:1]
            pos = argmax(second_terms .* x_same)
            loss_cur = nn .- second_terms[pos]
    
            mean_left_cur  = zeros(1,m)
            mean_right_cur = zeros(1,m)
            mean_left_cur[argmax(Cy[pos,:])] = 1
            mean_right_cur[argmax(RCy[pos,:])] = 1
            threshold_cur = (x_ordered[pos] + x_ordered[pos+1])/2
        else

            Ynan = y_ordered .!== NaN
            y_ordered .*= Ynan

            Nansum = cumsum(Ynan, dims = 1)[1:nn-1,:]
            NansumR = reverse(cumsum(Ynan[end:-1:1,:], dims = 1)[1:nn-1,:], dims = 1)
            Nansum += 1e-1 * (Nansum .== 0)
            NansumR += 1e-1 * (NansumR .== 0)
            
            Cy = cumsum(y_ordered, dims = 1)
            RCy = Cy[nn, :]' .- Cy
            y2 = sum(y_ordered.^2)
            
            Cy_norms = reshape(sum(Cy[1:nn-1,:].^2 ./Nansum, dims=2), nn-1)
            RCy_norms = reshape(sum(RCy[1:nn-1,:].^2 ./NansumR, dims=2), nn-1)

            second_terms = Cy_norms + RCy_norms
            pos = argmax(second_terms .* x_same)
        
            loss_cur = y2 - second_terms[pos]
            mean_left_cur = reshape(Cy[pos,:]./Nansum[pos,:], (1,m))
            mean_right_cur = reshape(RCy[pos,:]./NansumR[pos,:], (1,m))
            threshold_cur = (x_ordered[pos] + x_ordered[pos+1])/2
        end
    
        if loss_cur < loss_best
            loss_best = loss_cur
            threshold_best = threshold_cur
            mean_left_best = mean_left_cur
            mean_right_best = mean_right_cur
            label_best = ff  
        end
    end
    
    best_tree = Array{Any,1}(undef, 4)
    best_tree[1] = label_best
    best_tree[2] = threshold_best
    if D == 1
        if type == "R"
            best_tree[3] = mean_left_best
            best_tree[4] = mean_right_best
        else  # Classification
            best_tree[3] = vec(mean_left_best)
            best_tree[4] = vec(mean_right_best)
        end
        if type == "C"
            loss_best = sum((y - tree_eval(best_tree, X, D, m)) .> 1e-8 ) 
        end      
        return loss_best, best_tree
    else
        num_threshold = sum(X[:,label_best] .< threshold_best)
        od = sortperm(X[:,label_best])
        
        loss_left, best_left = greedy_tree(X[od[1:num_threshold],:], y[od[1:num_threshold],:], D-1, type)
        loss_right, best_right = greedy_tree(X[od[num_threshold+1:nn],:], y[od[num_threshold+1:nn],:], D-1, type)
        best_tree[3] = best_left
        best_tree[4] = best_right
        return loss_left + loss_right, best_tree
    end
end

## inv permutation
function inv_perm(a)
    n = size(a)[1]
    b = convert(Array{Int64,1}, zeros(n))
    for i=1:n
        b[a[i]] = i
    end
    return b
end

## quick findall in julia
function myfindall(f, a::Array{T, N}) where {T, N}
    j = 1
    b = Vector{Int}(undef, length(a))
    @inbounds for i in eachindex(a)
        b[j] = i
        j = ifelse(f(a[i]), j+1, j)
    end
    resize!(b, j-1)
    sizehint!(b, length(b))
    return b
end

## quick treesplit in julia
function treesplit(f, a::Array{T, N}) where {T, N}
    j1 = 1
    j2 = 1
    b1 = Vector{Int}(undef, length(a))
    b2 = Vector{Int}(undef, length(a))
    @inbounds for i in eachindex(a)
        b1[j1] = i
        b2[j2] = i
        if f(a[i])
            j1 += 1
        else
            j2 += 1
        end
    end
    resize!(b1, j1-1)
    sizehint!(b1, length(b1))
    resize!(b2, j2-1)
    sizehint!(b2, length(b2))
    return b1, b2
end

## quick fliter in julia
function myfliter(f, a::Array{T, N}) where {T, N}
    j = 1
    b = Vector{Int}(undef, length(a))
    @inbounds for i in eachindex(a)
        b[j] = a[i]
        j = ifelse(f(a[i]), j+1, j)
    end
    resize!(b, j-1)
    sizehint!(b, length(b))
    return b
end

## quick find interval in julia
function myfindinterval(f, a::Array{T, N}) where {T, N}

    b1 = Vector{Int}(undef, length(a))
    b2 = Vector{Int}(undef, length(a))
    flip = true
    cur = 1    
    @inbounds for i in eachindex(a)
        if f(a[i]) == flip
            if flip
                b1[cur] = i
                flip = false
            else
                b2[cur] = i-1
                flip = true
                cur += 1
            end
        end
    end
    if flip == false
        b2[cur] = length(a)
        cur += 1
    end
    resize!(b1, cur-1)
    resize!(b2, cur-1)
    sizehint!(b1, length(b1))
    sizehint!(b2, length(b2))
    return b1, b2
end
