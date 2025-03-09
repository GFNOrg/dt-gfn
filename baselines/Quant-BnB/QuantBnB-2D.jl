using LinearAlgebra
include("gen_data.jl")
include("lowerbound_middle.jl")
include("Algorithms.jl")

function QuantBnB_2D(X, y, s, ub, mid_method, mid_ratio, AL = nothing, treetype = "R", ifprint = false)
    println("QuantBnB_2D called with:")
    println("s = $s, ub = $ub, mid_method = $mid_method, mid_ratio = $mid_ratio")
    println("Input shape: X $(size(X)), y $(size(y))")

    n, p = size(X)
    if AL === nothing
        AL = Array{Any,2}(undef, p, 4)
        ff = [f for f=1:p]
        for k = 1:p
            AL[k, 1] = k
            AL[k, 2] = [1, n]
            AL[k, 3] = ff
            AL[k, 4] = ff
        end
    end

    global X_sort, X_sortperm, X_invperm, X_count, X_invcount, X_dict1, X_dict2, X_permcount
    X_sort, X_sortperm, X_invperm, X_count, X_invcount, X_dict1, X_dict2, X_permcount = presort(X)
    remain_tree = count_remain(AL)
    remain_interval = count_remain2(AL)
    num_loop = 0
    total_time = 0
    best_tree = 0

    if ifprint
        println("Total number of trees = ", remain_tree)
        println("Total number of intervals = ", remain_interval)
        println("--------------------------------------")
    end

    
    while remain_tree > 1 && num_loop < 20
        st = time()
            
        AL, ub, best_tree = screening_search_2D(X, y, s, AL, ub, best_tree, mid_method, mid_ratio, treetype)
        remain_tree = count_remain(AL)
        remain_interval = count_remain2(AL)
        time_loop = time() - st
        num_loop += 1
        total_time += time_loop

        if ifprint
            println("Loop ", num_loop)
            println("Number of remaining trees = ", remain_tree)
            println("Number of remaining intervals = ", remain_interval)
        
            println("time = ", time_loop)
            println("--------------------------------------")
        end
    end
    
    if ifprint
        println("Obj = ", ub)
        println("Tree is ", best_tree)
        println("total time = ", total_time)
    end

    println("QuantBnB_2D finished")
    println("Final ub = $ub")
    println("Best tree:")
    print_tree(best_tree)
    
    return ub, best_tree, AL
    
end

function count_remain(AL)
    
    value = 0
    n_rows = size(AL)[1]
    if n_rows == 0
        return 0
    end
    for i = 1:n_rows
        range = AL[i,2]
        fs_left = AL[i,3]
        fs_right = AL[i,4]    
        num_unique = length(unique(X_sort[range[1]:range[2], AL[i,1]])) 
        if num_unique == 0
            num_unique += 1
        end
        value += num_unique * (size(fs_left)[1] + size(fs_right)[1])
    end
    return value
end


function count_remain2(AL)
    
    value = 0
    n_rows = size(AL)[1]
    if n_rows == 0
        return 0
    end
    for i = 1:n_rows
        range = AL[i,2]
        fs_left = AL[i,3]
        fs_right = AL[i,4]  
        value +=  (size(fs_left)[1] + size(fs_right)[1])
    end
    return value
end



function screening_search_2D(X, y, s, AL, ub, best_tree, mid_method, mid_ratio, treetype)
    
    n, _ = size(X)
    quantiles = [i/s for i=0:s]
    n_rows = size(AL)[1]
    find_new_tree = 0
    ii = 0
    AL_new = Array{Any, 2}(undef, 0, 4)
    best_tree_label = Array{Any, 1}(undef, 4)
    LLBB = Array{Any, 2}(undef, n_rows, s)
    LLBB_left = Array{Any, 2}(undef, n_rows, s)
    LLBB_right = Array{Any, 2}(undef, n_rows, s)
    for i = 1:n_rows
    
        f0 = AL[i,1]
        range = AL[i,2]
        fs_left = AL[i,3]
        fs_right = AL[i,4]      
        
        n_left = size(fs_left)[1]
        n_right = size(fs_right)[1]
        order_f0 = X_sortperm[:,f0]
        
        count_s = X_count[range[1],f0] 
        count_e = X_count[range[2],f0] + 1
        unique_f0 = (count_e - count_s)
        part_f0 = X_invcount[f0][count_s:count_e]
        new_pos = count_s .+ convert(Array{Int64,1}, round.(quantiles .* unique_f0))
        spart_f0 = X_invcount[f0][new_pos]
        count_f0 = length(spart_f0) - 1           

        if unique_f0 <= s + 1
            llss_left = zeros(unique_f0+1, n_left)
            llss_right = zeros(unique_f0+1, n_right)            
            for t2 = 1:unique_f0+1
                for j = 1:n_left
                    llss_left[t2, j], _, _, _ = one_pass_search_mr(X, y, order_f0[1:part_f0[t2]], fs_left[j], treetype)
                end
                for j = 1:n_right
                    llss_right[t2, j], _, _, _ = one_pass_search_mr(X, y, order_f0[part_f0[t2]+1:n], fs_right[j], treetype)
                end

                LOSS = (llss_left[t2, :] .+ llss_right[t2, :]') 
                LOSS_min = minimum(LOSS)
                
                if LOSS_min < ub
                    LOSS_argmin = argmin(LOSS)   
                    find_new_tree = 1
                    ub = LOSS_min
                    best_tree_label[1] = f0
                    best_tree_label[2] = part_f0[t2]
                    best_tree_label[3] = fs_left[LOSS_argmin[1]]
                    best_tree_label[4] = fs_right[LOSS_argmin[2]]      
                end
            end
            
        else
            llss_left = zeros(count_f0, n_left)
            llss_right = zeros(count_f0, n_right)
            for t2 = 1:count_f0
                for j = 1:n_left
                    llss_left[t2, j], _, _, _ = one_pass_search_mr(X, y, order_f0[1:spart_f0[t2]], fs_left[j], treetype)
                end
            
                for j = 1:n_right
                    llss_right[t2, j], _, _, _ = one_pass_search_mr(X, y, order_f0[spart_f0[t2+1]+1:n], fs_right[j], treetype)
                end
                
                LB_MID = 0
                n_sub = spart_f0[t2+1] - spart_f0[t2]
                if n_sub > 2
                    total_compute = n * (count_f0 - 1)
                    s_mid =  Int(ceil(mid_ratio * total_compute / n_sub))
                    LB_MID = lower_bound_mid(X, y, f0, [spart_f0[t2]+1,spart_f0[t2+1]], fs_left, fs_right, s_mid, mid_method, treetype)
                end

                
                ## sum it up
                LLBB[i, t2] = (llss_left[t2, :] .+ llss_right[t2, :]') .+ LB_MID
                LLBB_left[i, t2] = minimum(LLBB[i, t2], dims = 2)
                LLBB_right[i, t2] = minimum(LLBB[i, t2], dims = 1)
                
                LLBB_left[i, t2] = reshape(LLBB_left[i, t2], maximum(size(LLBB_left[i, t2])) )
                LLBB_right[i, t2] = reshape(LLBB_right[i, t2], maximum(size(LLBB_right[i, t2])) )  
            end
            
            ## update ub and best tree label
            for j1 = 1:n_left
                for j2 = 1:n_right
                    LOSS = llss_left[2:count_f0,j1] .+ llss_right[1:count_f0-1,j2]
                    tmp = minimum(LOSS)
                    if tmp < ub
                        find_new_tree = 1
                        ub = tmp
                        tt = argmin(LOSS)
                        best_tree_label[1] = f0
                        best_tree_label[2] = spart_f0[tt+1]
                        best_tree_label[3] = fs_left[j1]
                        best_tree_label[4] = fs_right[j2]                            
                    end
                end
            end
        end
    end
    
    ## update best tree
    if find_new_tree == 1
        f0, split0 = best_tree_label[1], best_tree_label[2]
        f1, f2 = best_tree_label[3], best_tree_label[4]
        order_f0 = X_sortperm[:,f0]
        xx = X[order_f0, f0]
        l1, b1, lm1, rm1 = one_pass_search_mr(X, y, order_f0[1:split0], f1, treetype)
        l2, b2, lm2, rm2 = one_pass_search_mr(X, y, order_f0[split0+1:n], f2, treetype)
        ub = l1 + l2
        best_tree = Array{Any,1}(undef, 4)
        best_tree[1] = best_tree_label[1]
        if split0 == 0
            best_tree[2] = -1e20
        elseif split0 >= length(xx)
            best_tree[2] = 1e20
        else 
            best_tree[2] = 0.5*(xx[split0] + xx[split0+1])
        end
        best_tree[3] = Array{Any,1}(undef, 4)
        best_tree[3][1], best_tree[3][2], best_tree[3][3], best_tree[3][4] = f1, b1, lm1, rm1
        best_tree[4] = Array{Any,1}(undef, 4)
        best_tree[4][1], best_tree[4][2], best_tree[4][3], best_tree[4][4] = f2, b2, lm2, rm2
    end

    ## generate AL_new
    for i = 1:n_rows
    
        f0 = AL[i,1]
        range = AL[i,2]
        fs_left = AL[i,3]
        fs_right = AL[i,4]      
        
        order_f0 = X_sortperm[:,f0]
        
        count_s = X_count[range[1],f0] 
        count_e = X_count[range[2],f0] + 1
        unique_f0 = (count_e - count_s)
        new_pos = count_s .+ convert(Array{Int64,1}, round.(quantiles .* unique_f0))
        spart_f0 = X_invcount[f0][new_pos]
        count_f0 = length(spart_f0) - 1
        
        if unique_f0 > s + 1
            for t2 = 1:count_f0
                AL_left = findall(x -> x <= ub, LLBB_left[i, t2])
                AL_right = findall(x -> x <= ub, LLBB_right[i, t2])
                if size(AL_left)[1]>0 && size(AL_right)[1]>0
                    AL_new_row = Array{Any}(undef, 1, 4)
                    AL_new_row[1, 1] = AL[i][1]
                    AL_new_row[1, 2] = [spart_f0[t2]+1, spart_f0[t2+1]]
                    AL_new_row[1, 3] = fs_left[AL_left]
                    AL_new_row[1, 4] = fs_right[AL_right]
                    if ii == 0
                        AL_new = AL_new_row
                        ii = 1
                    else
                        AL_new = vcat(AL_new, AL_new_row)
                    end
                end
            end
        end
    end
    
    return AL_new, ub, best_tree
end

function exhaustive_screening(X, y, f0, range, fs_left, fs_right, treetype)
    n, p = size(X)
    ub = 1e20
    order_f0 = X_sortperm[:,f0]
    for t0 = range[1]:range[2]

        ls_lb = 1e20
        ls_rb = 1e20
        for j = 1:length(fs_left)
            loss_left, _, _, _ = one_pass_search_mr(X, y, order_f0[1:t0], fs_left[j], treetype)
            if loss_left < ls_lb
                ls_lb = loss_left
            end
        end
        for j = 1:length(fs_right)
            loss_right, _, _, _ = one_pass_search_mr(X, y, order_f0[t0+1:n], fs_right[j], treetype)
            if loss_right < ls_rb
                ls_rb = loss_right
            end
        end
        if ls_lb + ls_rb < ub
            ub = ls_lb + ls_rb
        end 
    end
    return ub
end
 









