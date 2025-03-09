using LinearAlgebra
include("gen_data.jl")
include("QuantBnB-2D.jl")
include("lowerbound_middle.jl")
include("Algorithms.jl")

function QuantBnB_3D(X, y, s, s2, ub, mid_method, mid_ratio, AALL = nothing, treetype = "R",
                         timelimit = 1e10)

    n, p = size(X)
    if AALL === nothing
        AL = Array{Any,2}(undef, p, 5)
        ff = [f for f=1:p]
        for k = 1:p
            AL[k, 1] = k
            AL[k, 2] = [1, n]
            AL[k, 3] = deepcopy(ff)
            AL[k, 4] = deepcopy(ff)
            AL[k, 5] = 1e20
        end
        AALL = Array{Any,2}(undef, p, 4)
        for k = 1:p
            AALL[k, 1] = k
            AALL[k, 2] = [1, n]
            AALL[k, 3] = deepcopy(AL)
            AALL[k, 4] = deepcopy(AL)
        end
    end

    global X_sort, X_sortperm, X_invperm, X_count, X_invcount, X_dict1, X_dict2, X_permcount
    X_sort, X_sortperm, X_invperm, X_count, X_invcount, X_dict1, X_dict2, X_permcount = presort(X)
    remain_tree = count_remain_3D(AALL)
    remain_interval = count_remain_3D2(AALL)
    num_loop = 0
    total_time = 0
    best_tree = 0
    println("Total number of trees = ", remain_tree)
    println("Total number of intervals = ", remain_interval)
    println("--------------------------------------")
    st = time()   
    
    while remain_tree > 5000 && num_loop < 20 && total_time < timelimit && remain_tree > remain_interval * 10
        st = time()
        AALL, ub, best_tree = screening_search_3D(X, y, s, s2, AALL,
                                                  ub, best_tree, mid_method, mid_ratio, treetype, timelimit - total_time)
        remain_tree = count_remain_3D(AALL)
        remain_interval = count_remain_3D2(AALL)
        time_loop = time() - st
        num_loop += 1
        total_time += time_loop
        println("Loop ", num_loop)
        println("Number of remaining trees = ", remain_tree)
        println("Total number of intervals = ", remain_interval)
        println("Current objective = ", ub)
        println("time = ", time_loop)
        println("--------------------------------------")
    end
    st = time()
    if remain_tree > 0 && total_time < timelimit
        ub, best_tree = exhaustive_search_3D(X, y, AALL, ub, best_tree, treetype, timelimit - total_time)
    end
    total_time += time() - st


    println("Obj = ", ub)
    println("Tree is ", best_tree)
    println("total time = ", total_time)
    return ub, best_tree
    
end

function exhaustive_search_3D(X, y, AALL, ub, best_tree, treetype, timelimit = 1e10)
    st = time()
    n_rows = size(AALL)[1]
    n, _ = size(X)
    if n_rows == 0
        return 0
    end
    for i = 1:n_rows   
        
        f0 = AALL[i,1]
        range = AALL[i,2]
        AL_left = AALL[i,3]
        AL_right = AALL[i,4]
        
        order_f0 = X_sortperm[:,f0]
        count_s = X_count[range[1],f0] 
        count_e = X_count[range[2],f0] + 1
        unique_f0 = (count_e - count_s)
        part_f0 = X_invcount[f0][count_s:count_e]
        if part_f0[1] == 0
            part_f0 = part_f0[2:end]
            unique_f0 -= 1
        end
        if part_f0[end] == n
            part_f0 = part_f0[1:end-1]
            unique_f0 -= 1
        end
        partend_f0 = part_f0
        partstart_f0 = part_f0 .+ 1
        counttotal_f0 = unique_f0 + 1

        for t2 = 1:counttotal_f0
            
            # left node
            idx_left = order_f0[1:partend_f0[t2]]
            n_left = length(idx_left)
            n_rows_left = size(AL_left)[1]
            llss_leftleft = Array{Any, 1}(undef, n_rows_left)
            llss_leftright = Array{Any, 1}(undef, n_rows_left)
            llss_leftstart = Array{Any, 1}(undef, n_rows_left)
            for i_left = 1:n_rows_left
                f1 = AL_left[i_left,1]
                range_left = AL_left[i_left,2]
                fs_left = AL_left[i_left,3]
                fs_right = AL_left[i_left,4] 
                num_fs_left = size(fs_left)[1]
                num_fs_right = size(fs_right)[1]  
                
                if n_left < n / 10
                    X_left = X[idx_left, f1]
                    y_left = y[idx_left,:]
                    od = sortperm(X_left)
                    x_ordered = X_left[od] 
                    y_ordered = y_left[od,:]
                    od_left = idx_left[od] 
                else
                    od = zeros(Int, n_left+1)
                    tmp = zeros(Int, n)
                    tmp[X_invperm[idx_left,f1]] = idx_left
                    count = 1
                    @inbounds for isort=1:n
                        od[count] = tmp[isort]
                        count += (tmp[isort] != zero(Int))
                    end
                    od_left = od[1:end-1]
                    x_ordered = X[od_left,f1]
                    y_ordered = y[od_left,:]
                end


                count_s1 = X_count[range_left[1],f1] 
                count_e1 = X_count[range_left[2],f1] + 1
                unique_f1 = (count_e1 - count_s1)
                part_f1 = X_invcount[f1][count_s1:count_e1]
                if part_f1[1] == 0
                    part_f1 = part_f1[2:end]
                    unique_f1 -= 1
                end
                if part_f1[end] == n
                    part_f1 = part_f1[1:end-1]
                    unique_f1 -= 1
                end

                part_total = Vector{Int64}()
                count_tmp = 1
                if part_f1[count_tmp] == 0
                    threshold = -1e20
                elseif part_f1[count_tmp] == n
                    threshold = 1e20
                else        
                    threshold = (X_sort[part_f1[count_tmp],f1]+X_sort[part_f1[count_tmp]+1,f1]) / 2
                end
                for t_left = 1:n_left
                    while x_ordered[t_left] > threshold
                        append!(part_total,t_left-1)
                        count_tmp += 1
                        if count_tmp <= unique_f1 + 1
                            if part_f1[count_tmp] == 0
                                threshold = -1e20
                            elseif part_f1[count_tmp] == n
                                threshold = 1e20
                            else        
                                threshold = (X_sort[part_f1[count_tmp],f1]+X_sort[part_f1[count_tmp]+1,f1]) / 2
                            end
                        else
                            threshold = 1e20
                        end
                    end
                end
                while count_tmp <= unique_f1 + 1
                    append!(part_total,n_left)
                    count_tmp += 1
                end
                partend_f1 = part_total
                partstart_f1 = part_total .+ 1
                counttotal_f1 = length(part_total) 


                llss_leftleft[i_left] = zeros(counttotal_f1, num_fs_left)
                llss_leftright[i_left] = zeros(counttotal_f1, num_fs_right)
                llss_leftstart[i_left] = part_f1

                for t2_left = 1:counttotal_f1
                    for j1 = 1:num_fs_left 
                        llss_leftleft[i_left][t2_left, j1], _, _, _ = one_pass_search_mr(X, y, od_left[1:partend_f1[t2_left]], fs_left[j1], treetype)
                    end
                    for j2 = 1:num_fs_right
                        llss_leftright[i_left][t2_left, j2], _, _, _ = one_pass_search_mr(X, y, od_left[partstart_f1[t2_left]:end], fs_right[j2], treetype)
                    end
                    if time() - st > timelimit
                        break
                    end   
                end
                if time() - st > timelimit
                    break
                end   
            end

            # right node
            idx_right = order_f0[partstart_f0[t2]:n]
            n_right = length(idx_right)
            n_rows_right = size(AL_right)[1]
            llss_rightleft = Array{Any, 1}(undef, n_rows_right)
            llss_rightright = Array{Any, 1}(undef, n_rows_right)
            llss_rightstart = Array{Any, 1}(undef, n_rows_right)
            for i_right = 1:n_rows_right
                f2 = AL_right[i_right,1]
                range_right = AL_right[i_right,2]
                fs_left = AL_right[i_right,3]
                fs_right = AL_right[i_right,4] 
                num_fs_left = size(fs_left)[1]
                num_fs_right = size(fs_right)[1]  
                
                if n_right < n / 10
                    X_right = X[idx_right, f2]
                    y_right = y[idx_right,:]
                    od = sortperm(X_right)
                    x_ordered = X_right[od] 
                    y_ordered = y_right[od,:]
                    od_right = idx_right[od] 
                else
                    od = zeros(Int, n_right+1)
                    tmp = zeros(Int, n)
                    tmp[X_invperm[idx_right,f2]] = idx_right
                    count = 1
                    @inbounds for isort=1:n
                        od[count] = tmp[isort]
                        count += (tmp[isort] != zero(Int))
                    end
                    od_right = od[1:end-1]
                    x_ordered = X[od_right,f2]
                    y_ordered = y[od_right,:]
                end

                count_s2 = X_count[range_right[1],f2] 
                count_e2 = X_count[range_right[2],f2] + 1
                unique_f2 = (count_e2 - count_s2)
                part_f2 = X_invcount[f2][count_s2:count_e2]
                if part_f2[1] == 0
                    part_f2 = part_f2[2:end]
                    unique_f2 -= 1
                end
                if part_f2[end] == n
                    part_f2 = part_f2[1:end-1]
                    unique_f2 -= 1
                end

                part_total = Vector{Int64}()
                count_tmp = 1
                if part_f2[count_tmp] == 0
                    threshold = -1e20
                elseif part_f2[count_tmp] == n
                    threshold = 1e20
                else        
                    threshold = (X_sort[part_f2[count_tmp],f2]+X_sort[part_f2[count_tmp]+1,f2]) / 2
                end
                for t_right = 1:n_right
                    while x_ordered[t_right] > threshold
                        append!(part_total,t_right-1)
                        count_tmp += 1
                        if count_tmp <= unique_f2 + 1
                            if part_f2[count_tmp] == 0
                                threshold = -1e20
                            elseif part_f2[count_tmp] == n
                                threshold = 1e20
                            else        
                                threshold = (X_sort[part_f2[count_tmp],f2]+X_sort[part_f2[count_tmp]+1,f2]) / 2
                            end
                        else
                            threshold = 1e20
                        end
                    end
                end
                while count_tmp <= unique_f2 + 1
                    append!(part_total,n_right)
                    count_tmp += 1
                end
                partend_f2 = part_total
                partstart_f2 = part_total .+ 1
                counttotal_f2 = length(part_total) 

                llss_rightleft[i_right] = zeros(counttotal_f2, num_fs_left)
                llss_rightright[i_right] = zeros(counttotal_f2, num_fs_right)
                llss_rightstart[i_right] = part_f2
                for t2_right = 1:counttotal_f2
                    for j1 = 1:num_fs_left 
                        llss_rightleft[i_right][t2_right, j1], _, _, _ = one_pass_search_mr(X, y, od_right[1:partend_f2[t2_right]], fs_left[j1], treetype)
                    end
                    for j2 = 1:num_fs_right
                        llss_rightright[i_right][t2_right, j2], _, _, _ = one_pass_search_mr(X, y, od_right[partstart_f2[t2_right]:end], fs_right[j2], treetype)
                    end
                    if time() - st > timelimit
                        break
                    end   
                end
                if time() - st > timelimit
                    break
                end   
            end
            if time() - st > timelimit
                break
            end    

            ### get upper bound
            llss_left = zeros(n_rows_left)
            llss_right = zeros(n_rows_right)
            for i_left = 1:n_rows_left
                llss_left[i_left] = minimum(minimum(llss_leftleft[i_left],dims = 2) + minimum(llss_leftright[i_left],dims = 2))
            end
            for i_right = 1:n_rows_right
                llss_right[i_right] = minimum(minimum(llss_rightleft[i_right],dims = 2) + minimum(llss_rightright[i_right],dims = 2))
            end

            if minimum(llss_left) + minimum(llss_right) < ub - 1e-8
                ub = minimum(llss_left) + minimum(llss_right)
                where_left = argmin(llss_left)
                where_right = argmin(llss_right)

                split0 = partend_f0[t2]
                f1 = AL_left[where_left, 1]
                f2 = AL_right[where_right, 1]
                t2_left = argmin(minimum(llss_leftleft[where_left],dims = 2) + minimum(llss_leftright[where_left],dims = 2))[1]
                t2_right = argmin(minimum(llss_rightleft[where_right],dims = 2) + minimum(llss_rightright[where_right],dims = 2))[1]
                f11 = AL_left[where_left, 3][argmin(llss_leftleft[where_left][t2_left,:])]
                f12 = AL_left[where_left, 4][argmin(llss_leftright[where_left][t2_left,:])]
                f21 = AL_right[where_right, 3][argmin(llss_rightleft[where_right][t2_right,:])]
                f22 = AL_right[where_right, 4][argmin(llss_rightright[where_right][t2_right,:])]
                s1_tmp = llss_leftstart[where_left][t2_left]
                s2_tmp = llss_rightstart[where_right][t2_right]

                if s1_tmp == 0
                    split1 = -1e20
                elseif s1_tmp == n
                    split1 = 1e20
                else
                    split1 = (X_sort[s1_tmp,f1] + X_sort[s1_tmp+1,f1]) / 2
                end
                if s2_tmp == 0
                    split2 = -1e20
                elseif s2_tmp == n
                    split2 = 1e20
                else
                    split2 = (X_sort[s2_tmp,f2] + X_sort[s2_tmp+1,f2]) / 2
                end
                x_ordered = X_sort[:,f0]
                order_f0 = X_sortperm[:,f0]
                idx_left = order_f0[1:split0]
                idx_LL = idx_left[findall(x -> x < split1, X[idx_left, f1])]
                idx_LR =  idx_left[findall(x -> x >= split1, X[idx_left, f1])]
                idx_right = order_f0[split0+1:n]
                idx_RL = idx_right[findall(x -> x < split2, X[idx_right, f2])]
                idx_RR =  idx_right[findall(x -> x >= split2, X[idx_right, f2])]
                l11, b11, lm11, rm11 = one_pass_search_mr(X, y, idx_LL, f11, treetype)
                l12, b12, lm12, rm12 = one_pass_search_mr(X, y, idx_LR, f12, treetype)
                l21, b21, lm21, rm21 = one_pass_search_mr(X, y, idx_RL, f21, treetype)
                l22, b22, lm22, rm22 = one_pass_search_mr(X, y, idx_RR, f22, treetype)
                if abs(ub - (l11 + l12 + l21 + l22)) > 1e-8
                    println(ub)
                    println(l11+l12+l21+l22)
                    sleep(0.1)
                    #throw(ErrorException)
                end
                best_tree = Array{Any,1}(undef, 4)
                best_tree[1] = f0
                best_tree[2] = 0.5 * (x_ordered[split0] + x_ordered[split0+1])
                best_tree[3] = Array{Any,1}(undef, 4)
                best_tree[3][1], best_tree[3][2] = f1, split1
                best_tree[4] = Array{Any,1}(undef, 4)
                best_tree[4][1], best_tree[4][2] = f2, split2
                best_tree[3][3] = Array{Any,1}(undef, 4)
                best_tree[3][4] = Array{Any,1}(undef, 4)
                best_tree[4][3] = Array{Any,1}(undef, 4)
                best_tree[4][4] = Array{Any,1}(undef, 4)
                best_tree[3][3][1], best_tree[3][3][2], best_tree[3][3][3], best_tree[3][3][4] = f11, b11, lm11, rm11
                best_tree[3][4][1], best_tree[3][4][2], best_tree[3][4][3], best_tree[3][4][4] = f12, b12, lm12, rm12
                best_tree[4][3][1], best_tree[4][3][2], best_tree[4][3][3], best_tree[4][3][4] = f21, b21, lm21, rm21
                best_tree[4][4][1], best_tree[4][4][2], best_tree[4][4][3], best_tree[4][4][4] = f22, b22, lm22, rm22

            end  
                   
        end
        if time() - st > timelimit
            break
        end   
    end
    return ub, best_tree
end


function count_remain_3D(AALL)
    
    value = 0
    n_rows = size(AALL)[1]
    if n_rows == 0
        return 0
    end
    for i = 1:n_rows
        range = AALL[i,2]
        AL_left = AALL[i,3]
        AL_right = AALL[i,4]    
        num_unique = length(unique(X_sort[range[1]:range[2], AALL[i,1]])) 
        if num_unique == 0
            num_unique += 1
        end
        value += num_unique * (count_remain(AL_left) + count_remain(AL_right))
    end
    return value
end

function count_remain_3D2(AALL)
    
    value = 0
    n_rows = size(AALL)[1]
    if n_rows == 0
        return 0
    end
    for i = 1:n_rows
        AL_left = AALL[i,3]
        AL_right = AALL[i,4]    
        value += (count_remain2(AL_left) + count_remain2(AL_right))
    end
    return value
end
    

function screening_search_3D(X, y, s, s2, AALL, ub, best_tree, mid_method, mid_ratio, treetype, timelimit)
    
    st = time()
    n, _ = size(X)
    quantiles = [i/s for i=0:s]
    quantiles2 = [i/s2 for i=0:s2]
    n_rows = size(AALL)[1]
    find_new_tree = 0
    i_root = 0
    AALL_new = Array{Any, 2}(undef, 0, 4)
    best_tree_label = Array{Any, 1}(undef, 4)

    for i = 1:n_rows   
        
        f0 = AALL[i,1]
        range = AALL[i,2]
        AL_left = AALL[i,3]
        AL_right = AALL[i,4]
        
        order_f0 = X_sortperm[:,f0]
        count_s = X_count[range[1],f0] 
        count_e = X_count[range[2],f0] + 1
        unique_f0 = (count_e - count_s)
        part_f0 = X_invcount[f0][count_s:count_e]
        new_pos = count_s .+ convert(Array{Int64,1}, round.(quantiles .* unique_f0))
        spart_f0 = X_invcount[f0][new_pos]
        count_f0 = length(spart_f0) - 1     

        
        if unique_f0 <= s + 1
            if part_f0[1] == 0
                part_f0 = part_f0[2:end]
                unique_f0 -= 1
            end
            if part_f0[end] == n
                part_f0 = part_f0[1:end-1]
                unique_f0 -= 1
            end
            partend_f0 = part_f0
            partstart_f0 = part_f0 .+ 1
            counttotal_f0 = unique_f0 + 1
            exh_f0 = 1
        else
            partend_f0 = spart_f0[1:count_f0] 
            partstart_f0 = spart_f0[2:count_f0+1] .+ 1
            counttotal_f0 = count_f0
            exh_f0 = 0
        end

        UPPER_left = ones(counttotal_f0) .* 1e20
        UPPER_right = ones(counttotal_f0) .* 1e20
        UPPER_LL = zeros(counttotal_f0) 
        UPPER_LR = zeros(counttotal_f0) 
        UPPER_RL = zeros(counttotal_f0) 
        UPPER_RR = zeros(counttotal_f0) 
        UPPER_f1min = zeros(Int, counttotal_f0) 
        UPPER_f1split = zeros(counttotal_f0) 
        UPPER_f2min = zeros(Int, counttotal_f0) 
        UPPER_f2split = zeros(counttotal_f0) 

        llss_left = Array{Any, 1}(undef, counttotal_f0)
        llss_leftleft = Array{Any, 1}(undef, counttotal_f0)
        llss_leftright = Array{Any, 1}(undef, counttotal_f0)
        llss_leftleft2 = Array{Any, 1}(undef, counttotal_f0)
        llss_leftright2 = Array{Any, 1}(undef, counttotal_f0)
        llss_leftleft3 = Array{Any, 1}(undef, counttotal_f0)
        llss_leftright3 = Array{Any, 1}(undef, counttotal_f0)
        llss_leftstart = Array{Any, 1}(undef, counttotal_f0)
        LB_leftMID = Array{Any, 1}(undef, counttotal_f0)
        exh_f1 = Array{Any, 1}(undef, counttotal_f0)
        issingle_f1 = Array{Any, 1}(undef, counttotal_f0)

        llss_right = Array{Any, 1}(undef, counttotal_f0)
        llss_rightleft = Array{Any, 1}(undef, counttotal_f0)
        llss_rightright = Array{Any, 1}(undef, counttotal_f0)
        llss_rightleft2 = Array{Any, 1}(undef, counttotal_f0)
        llss_rightright2 = Array{Any, 1}(undef, counttotal_f0)
        llss_rightleft3 = Array{Any, 1}(undef, counttotal_f0)
        llss_rightright3 = Array{Any, 1}(undef, counttotal_f0)
        llss_rightstart = Array{Any, 1}(undef, counttotal_f0)
        LB_rightMID = Array{Any, 1}(undef, counttotal_f0)
        exh_f2 = Array{Any, 1}(undef, counttotal_f0)
        issingle_f2 = Array{Any, 1}(undef, counttotal_f0)

        llss_mid = zeros(counttotal_f0, size(AL_left)[1], size(AL_right)[1])

        for t2 = 1:counttotal_f0

            # left node
            idx_left = order_f0[1:partend_f0[t2]]
            n_left = length(idx_left)
            n_rows_left = size(AL_left)[1]
    
            llss_left[t2] = zeros(n_rows_left)
            llss_leftleft[t2] = Array{Any, 1}(undef, n_rows_left)
            llss_leftright[t2] = Array{Any, 1}(undef, n_rows_left)
            llss_leftleft2[t2] = Array{Any, 1}(undef, n_rows_left)
            llss_leftright2[t2] = Array{Any, 1}(undef, n_rows_left)
            llss_leftleft3[t2] = Array{Any, 1}(undef, n_rows_left)
            llss_leftright3[t2] = Array{Any, 1}(undef, n_rows_left)
            llss_leftstart[t2] = Array{Any, 1}(undef, n_rows_left)
            LB_leftMID[t2] = Array{Any, 1}(undef, n_rows_left)
            exh_f1[t2] = zeros(n_rows_left)
            issingle_f1[t2] = zeros(n_rows_left)

            for i_left = 1:n_rows_left
                f1 = AL_left[i_left,1]
                range_left = AL_left[i_left,2]
                fs_left = AL_left[i_left,3]
                fs_right = AL_left[i_left,4] 
                num_fs_left = size(fs_left)[1]
                num_fs_right = size(fs_right)[1]  
                
                if n_left < n / 10
                    X_left = X[idx_left, f1]
                    y_left = y[idx_left,:]
                    od = sortperm(X_left)
                    x_ordered = X_left[od] 
                    y_ordered = y_left[od,:]
                    od_left = idx_left[od] 
                else
                    od = zeros(Int, n_left+1)
                    tmp = zeros(Int, n)
                    tmp[X_invperm[idx_left,f1]] = idx_left
                    count = 1
                    @inbounds for isort=1:n
                        od[count] = tmp[isort]
                        count += (tmp[isort] != zero(Int))
                    end
                    od_left = od[1:end-1]
                    x_ordered = X[od_left,f1]
                    y_ordered = y[od_left,:]
                end


                count_s1 = X_count[range_left[1],f1] 
                count_e1 = X_count[range_left[2],f1] + 1
                unique_f1 = (count_e1 - count_s1)
                part_f1 = X_invcount[f1][count_s1:count_e1]
                new_pos = count_s1 .+ convert(Array{Int64,1}, round.(quantiles2 .* unique_f1))
                spart_f1 = X_invcount[f1][new_pos]
                count_f1 = length(spart_f1) - 1 
                partend_f1, partstart_f1, counttotal_f1 = 0, 0, 0
                if range_left[1] > range_left[2]
                    if range_left[1] > n || range_left[2] == 0
                        throw(ErrorException)
                    end
                    exh_f1[t2][i_left] = 1
                    issingle_f1[t2][i_left] = 1
                    llss_leftstart[t2][i_left] = part_f1
                    threshold = (X_sort[part_f1[1],f1]+X_sort[part_f1[1]+1,f1]) / 2
                    if n_left == 0
                        partend_f1 = [0]
                        partstart_f1 = partend_f1 .+ 1
                    else
                        partend_f1 = [sum(x_ordered.<threshold)]
                        partstart_f1 = partend_f1 .+ 1
                    end
                    counttotal_f1 = 1
                elseif unique_f1 <= s2 + 1
                    exh_f1[t2][i_left] = 1
                    issingle_f1[t2][i_left] = 0
                    if part_f1[1] == 0
                        part_f1 = part_f1[2:end]
                    end
                    if part_f1[end] == n
                        part_f1 = part_f1[1:end-1]
                    end
                    count_f1 = length(part_f1) - 1

                    llss_leftstart[t2][i_left] = part_f1
                    part_total = Vector{Int64}()
                    count_tmp = 1
                    if part_f1[count_tmp] == 0
                        threshold = -1e20
                    elseif part_f1[count_tmp] == n
                        threshold = 1e20
                    else        
                        threshold = (X_sort[part_f1[count_tmp],f1]+X_sort[part_f1[count_tmp]+1,f1]) / 2
                    end
                    for t_left = 1:n_left
                        while x_ordered[t_left] > threshold
                            append!(part_total,t_left-1)
                            count_tmp += 1
                            if count_tmp <= count_f1 + 1
                                if part_f1[count_tmp] == 0
                                    threshold = -1e20
                                elseif part_f1[count_tmp] == n
                                    threshold = 1e20
                                else        
                                    threshold = (X_sort[part_f1[count_tmp],f1]+X_sort[part_f1[count_tmp]+1,f1]) / 2
                                end
                            else
                                threshold = 1e20
                            end
                        end
                    end
                    while count_tmp <= count_f1 + 1
                        append!(part_total,n_left)
                        count_tmp += 1
                    end

                    partend_f1 = part_total
                    partstart_f1 = part_total .+ 1
                    counttotal_f1 = length(part_total) 
                else
                    exh_f1[t2][i_left] = 0
                    issingle_f1[t2][i_left] = 0
                    llss_leftstart[t2][i_left] = spart_f1
                    part_total = Vector{Int64}()
                    count_tmp = 1
                    if spart_f1[count_tmp] == 0
                        threshold = -1e20
                    elseif spart_f1[count_tmp] == n
                        threshold = 1e20
                    else        
                        threshold = (X_sort[spart_f1[count_tmp],f1]+X_sort[spart_f1[count_tmp]+1,f1]) / 2
                    end
                    for t_left = 1:n_left
                        while x_ordered[t_left] > threshold
                            append!(part_total,t_left-1)
                            count_tmp += 1
                            if count_tmp <= count_f1 + 1
                                if spart_f1[count_tmp] == 0
                                    threshold = -1e20
                                elseif spart_f1[count_tmp] == n
                                    threshold = 1e20
                                else        
                                    threshold = (X_sort[spart_f1[count_tmp],f1]+X_sort[spart_f1[count_tmp]+1,f1]) / 2
                                end
                            else
                                threshold = 1e20
                            end
                        end
                    end
                    while count_tmp <= count_f1 + 1
                        append!(part_total,n_left)
                        count_tmp += 1
                    end
                    counttotal_f1 = length(part_total) - 1
                    partend_f1 = part_total[1:counttotal_f1] 
                    partstart_f1 = part_total[2:counttotal_f1+1] .+ 1
                end

                llss_leftleft[t2][i_left] = zeros(counttotal_f1, num_fs_left)
                llss_leftright[t2][i_left] = zeros(counttotal_f1, num_fs_right)
                llss_leftleft2[t2][i_left] = zeros(counttotal_f1)
                llss_leftright2[t2][i_left] = zeros(counttotal_f1)
                llss_leftleft3[t2][i_left] = zeros(counttotal_f1, num_fs_left)
                llss_leftright3[t2][i_left] = zeros(counttotal_f1, num_fs_right)
                LB_leftMID[t2][i_left] = zeros(counttotal_f1, num_fs_left, num_fs_right)



                for t2_left = 1:counttotal_f1
                    for j1 = 1:num_fs_left 
                        llss_leftleft[t2][i_left][t2_left, j1], _, _, _ = one_pass_search_mr(X, y, od_left[1:partend_f1[t2_left]], fs_left[j1], treetype)
                    end
                    llss_leftleft2[t2][i_left][t2_left] = minimum(llss_leftleft[t2][i_left][t2_left, :])
                    for j2 = 1:num_fs_right
                        llss_leftright[t2][i_left][t2_left, j2], _, _, _ = one_pass_search_mr(X, y, od_left[partstart_f1[t2_left]:end], fs_right[j2], treetype)
                    end
                    llss_leftright2[t2][i_left][t2_left] = minimum(llss_leftright[t2][i_left][t2_left, :])
                    total_compute = n_left * (counttotal_f1 - 1)
                    n_sub = partstart_f1[t2_left] - partend_f1[t2_left] 
                    s_mid =  Int(ceil(mid_ratio * total_compute / n_sub))
                    if exh_f1[t2][i_left] == 0
                        LB_leftMID[t2][i_left][t2_left,:,:] = lower_bound_mid2(X, y, f1, od_left[partend_f1[t2_left]+1:partstart_f1[t2_left]-1],
                                    [llss_leftstart[t2][i_left][t2_left]+1,llss_leftstart[t2][i_left][t2_left+1]], fs_left, fs_right, s_mid, mid_method, treetype)
                    else
                        LB_leftMID[t2][i_left][t2_left,:,:] = zeros(num_fs_left, num_fs_right)
                    end
                    llss_leftleft3[t2][i_left][t2_left,:] = minimum(LB_leftMID[t2][i_left][t2_left,:,:] .+ llss_leftleft[t2][i_left][t2_left, :] .+ llss_leftright[t2][i_left][t2_left, :]', dims = 2)
                    llss_leftright3[t2][i_left][t2_left,:] = minimum(LB_leftMID[t2][i_left][t2_left,:,:] .+ llss_leftleft[t2][i_left][t2_left, :] .+ llss_leftright[t2][i_left][t2_left, :]', dims = 1)
                end
                if counttotal_f1 > 0
                    llss_tmp = zeros(num_fs_left, num_fs_right)
                    for j1 = 1:num_fs_left
                        for j2 = 1:num_fs_right
                            llss_tmp[j1,j2] = minimum(LB_leftMID[t2][i_left][:,j1,j2]+llss_leftleft[t2][i_left][:,j1]+llss_leftright[t2][i_left][:,j2])
                        end
                    end
                    llss_left[t2][i_left] = minimum(llss_tmp)
                else
                    llss_left[t2][i_left] = 0

                end
            end


            # right node
            idx_right = order_f0[partstart_f0[t2]:n]
            n_right = length(idx_right)
            n_rows_right = size(AL_right)[1]
    
            llss_right[t2] = zeros(n_rows_right)
            llss_rightleft[t2] = Array{Any, 1}(undef, n_rows_right)
            llss_rightright[t2] = Array{Any, 1}(undef, n_rows_right)
            llss_rightleft2[t2] = Array{Any, 1}(undef, n_rows_right)
            llss_rightright2[t2] = Array{Any, 1}(undef, n_rows_right)
            llss_rightleft3[t2] = Array{Any, 1}(undef, n_rows_right)
            llss_rightright3[t2] = Array{Any, 1}(undef, n_rows_right)
            llss_rightstart[t2] = Array{Any, 1}(undef, n_rows_right)
            LB_rightMID[t2] = Array{Any, 1}(undef, n_rows_right)
            exh_f2[t2] = zeros(n_rows_right)
            issingle_f2[t2] = zeros(n_rows_right)

            for i_right = 1:n_rows_right
                f2 = AL_right[i_right,1]
                range_right = AL_right[i_right,2]
                fs_left = AL_right[i_right,3]
                fs_right = AL_right[i_right,4] 
                num_fs_left = size(fs_left)[1]
                num_fs_right = size(fs_right)[1]  
                
                if n_right < n / 10
                    X_right = X[idx_right, f2]
                    y_right = y[idx_right,:]
                    od = sortperm(X_right)
                    x_ordered = X_right[od] 
                    y_ordered = y_right[od,:]
                    od_right = idx_right[od] 
                else
                    od = zeros(Int, n_right+1)
                    tmp = zeros(Int, n)
                    tmp[X_invperm[idx_right,f2]] = idx_right
                    count = 1
                    @inbounds for isort=1:n
                        od[count] = tmp[isort]
                        count += (tmp[isort] != zero(Int))
                    end
                    od_right = od[1:end-1]
                    x_ordered = X[od_right,f2]
                    y_ordered = y[od_right,:]
                end

                count_s2 = X_count[range_right[1],f2] 
                count_e2 = X_count[range_right[2],f2] + 1
                unique_f2 = (count_e2 - count_s2)
                part_f2 = X_invcount[f2][count_s2:count_e2]
                new_pos = count_s2 .+ convert(Array{Int64,1}, round.(quantiles2 .* unique_f2))
                spart_f2 = X_invcount[f2][new_pos]
                count_f2 = length(spart_f2) - 1 
                
                partend_f2, partstart_f2, counttotal_f2 = 0, 0, 0
                if range_right[1] > range_right[2]
                    if range_right[1] > n || range_right[2] == 0
                        throw(ErrorException)
                    end
                    exh_f2[t2][i_right] = 1
                    issingle_f2[t2][i_right] = 1
                    llss_rightstart[t2][i_right] = part_f2
                    threshold = (X_sort[part_f2[1],f2]+X_sort[part_f2[1]+1,f2]) / 2
                    if n_right == 0
                        partend_f2 = [0]
                        partstart_f2 = partend_f2 .+ 1
                    else
                        partend_f2 = [sum(x_ordered.<threshold)]
                        partstart_f2 = partend_f2 .+ 1
                    end
                    counttotal_f2 = 1
                elseif unique_f2 <= s2 + 1
                    exh_f2[t2][i_right] = 1
                    issingle_f2[t2][i_right] = 0
                    if part_f2[1] == 0
                        part_f2 = part_f2[2:end]
                    end
                    if part_f2[end] == n
                        part_f2 = part_f2[1:end-1]
                    end
                    count_f2 = length(part_f2) - 1

                    llss_rightstart[t2][i_right] = part_f2
                    part_total = Vector{Int64}()
                    count_tmp = 1
                    if part_f2[count_tmp] == 0
                        threshold = -1e20
                    elseif part_f2[count_tmp] == n
                        threshold = 1e20
                    else        
                        threshold = (X_sort[part_f2[count_tmp],f2]+X_sort[part_f2[count_tmp]+1,f2]) / 2
                    end
                    for t_right = 1:n_right
                        while x_ordered[t_right] > threshold
                            append!(part_total,t_right-1)
                            count_tmp += 1
                            if count_tmp <= count_f2 + 1
                                if part_f2[count_tmp] == 0
                                    threshold = -1e20
                                elseif part_f2[count_tmp] == n
                                    threshold = 1e20
                                else        
                                    threshold = (X_sort[part_f2[count_tmp],f2]+X_sort[part_f2[count_tmp]+1,f2]) / 2
                                end
                            else
                                threshold = 1e20
                            end
                        end
                    end
                    while count_tmp <= count_f2 + 1
                        append!(part_total,n_right)
                        count_tmp += 1
                    end

                    partend_f2 = part_total
                    partstart_f2 = part_total .+ 1
                    counttotal_f2 = length(part_total) 
                else
                    exh_f2[t2][i_right] = 0
                    issingle_f2[t2][i_right] = 0
                    part_total = Vector{Int64}()
                    llss_rightstart[t2][i_right] = spart_f2
                    count_tmp = 1
                    if spart_f2[count_tmp] == 0
                        threshold = -1e20
                    elseif spart_f2[count_tmp] == n
                        threshold = 1e20
                    else        
                        threshold = (X_sort[spart_f2[count_tmp],f2]+X_sort[spart_f2[count_tmp]+1,f2]) / 2
                    end
                    for t_right = 1:n_right
                        while x_ordered[t_right] > threshold
                            append!(part_total,t_right-1)
                            count_tmp += 1
                            if count_tmp <= count_f2 + 1
                                if spart_f2[count_tmp] == 0
                                    threshold = -1e20
                                elseif spart_f2[count_tmp] == n
                                    threshold = 1e20
                                else        
                                    threshold = (X_sort[spart_f2[count_tmp],f2]+X_sort[spart_f2[count_tmp]+1,f2]) / 2
                                end
                            else
                                threshold = 1e20
                            end
                        end
                    end
                    while count_tmp <= count_f2 + 1
                        append!(part_total,n_right)
                        count_tmp += 1
                    end

                    counttotal_f2 = length(part_total) - 1
                    partend_f2 = part_total[1:counttotal_f2] 
                    partstart_f2 = part_total[2:counttotal_f2+1] .+ 1
                end

                llss_rightleft[t2][i_right] = zeros(counttotal_f2, num_fs_left)
                llss_rightright[t2][i_right] = zeros(counttotal_f2, num_fs_right)
                llss_rightleft2[t2][i_right] = zeros(counttotal_f2)
                llss_rightright2[t2][i_right] = zeros(counttotal_f2)
                llss_rightleft3[t2][i_right] = zeros(counttotal_f2, num_fs_left)
                llss_rightright3[t2][i_right] = zeros(counttotal_f2, num_fs_right)
                LB_rightMID[t2][i_right] = zeros(counttotal_f2, num_fs_left, num_fs_right)

                

                for t2_right = 1:counttotal_f2
                    for j1 = 1:num_fs_left
                        llss_rightleft[t2][i_right][t2_right, j1], _, _, _ = one_pass_search_mr(X, y, od_right[1:partend_f2[t2_right]], fs_left[j1], treetype)
                    end
                    llss_rightleft2[t2][i_right][t2_right] = minimum(llss_rightleft[t2][i_right][t2_right, :])
                    for j2 = 1:num_fs_right
                        llss_rightright[t2][i_right][t2_right, j2], _, _, _ = one_pass_search_mr(X, y, od_right[partstart_f2[t2_right]:end], fs_right[j2], treetype)
                    end
                    llss_rightright2[t2][i_right][t2_right] = minimum(llss_rightright[t2][i_right][t2_right, :])
                    total_compute = n_right * (counttotal_f2 - 1)
                    n_sub = partstart_f2[t2_right] - partend_f2[t2_right] 
                    s_mid =  Int(ceil(mid_ratio * total_compute / n_sub))
                    if exh_f2[t2][i_right] == 0
                        LB_rightMID[t2][i_right][t2_right,:,:] = lower_bound_mid2(X, y, f2, od_right[partend_f2[t2_right]+1:partstart_f2[t2_right]-1],
                                    [llss_rightstart[t2][i_right][t2_right]+1,llss_rightstart[t2][i_right][t2_right+1]], fs_left, fs_right, s_mid, mid_method, treetype)
                    else
                        LB_rightMID[t2][i_right][t2_right,:,:] = zeros(num_fs_left, num_fs_right)
                    end
                    llss_rightleft3[t2][i_right][t2_right,:] = minimum(LB_rightMID[t2][i_right][t2_right,:,:] .+ llss_rightleft[t2][i_right][t2_right, :] .+ llss_rightright[t2][i_right][t2_right, :]', dims = 2)
                    llss_rightright3[t2][i_right][t2_right,:] = minimum(LB_rightMID[t2][i_right][t2_right,:,:] .+ llss_rightleft[t2][i_right][t2_right, :] .+ llss_rightright[t2][i_right][t2_right, :]', dims = 1)
                end
                if counttotal_f2 > 0
                    llss_tmp = zeros(num_fs_left, num_fs_right)
                    for j1 = 1:num_fs_left
                        for j2 = 1:num_fs_right
                            llss_tmp[j1,j2] = minimum(LB_rightMID[t2][i_right][:,j1,j2]+llss_rightleft[t2][i_right][:,j1]+llss_rightright[t2][i_right][:,j2])
                        end
                    end
                    llss_right[t2][i_right] = minimum(llss_tmp)
                else
                    llss_right[t2][i_right] = 0
                end
            end

            total_compute = n * (counttotal_f0 - 1)^2
            n_sub = partstart_f0[t2] - partend_f0[t2] 
            #s_mid =  Int(ceil(sqrt(mid_ratio * total_compute / n_sub / 10)))
            s_mid = 0 
            llss_mid[t2, :, :] = lower_bound_mid_3D(X, y, f0, [partend_f0[t2]+1, partstart_f0[t2]-1], AL_left, AL_right, s_mid, mid_method, mid_ratio, treetype)


            ### get upper bound
            minimum_left = zeros(n_rows_left)
            where_minleft = zeros(Int, n_rows_left)
            for i_left = 1:n_rows_left
                if exh_f1[t2][i_left] == 0
                    where_minleft[i_left] = argmin(llss_leftleft2[t2][i_left][2:end] .+ llss_leftright2[t2][i_left][1:end-1])
                    minimum_left[i_left] = minimum(llss_leftleft2[t2][i_left][2:end] .+ llss_leftright2[t2][i_left][1:end-1])
                else
                    where_minleft[i_left] = argmin(llss_leftleft2[t2][i_left] .+ llss_leftright2[t2][i_left])
                    minimum_left[i_left] = minimum(llss_leftleft2[t2][i_left] .+ llss_leftright2[t2][i_left])
                end
            end
            
            if minimum(minimum_left) < UPPER_left[t2]
                UPPER_left[t2] = minimum(minimum_left)
                where_upper = argmin(minimum_left)
                f1min = AL_left[where_upper,1]
                UPPER_f1min[t2] = f1min
                idx_tmp = 0
                if exh_f1[t2][where_upper] == 0
                    idx_tmp = where_minleft[where_upper] + 1
                else
                    idx_tmp = where_minleft[where_upper] 
                end
                if llss_leftstart[t2][where_upper][idx_tmp] == 0
                    UPPER_f1split[t2] = -1e20
                elseif llss_leftstart[t2][where_upper][idx_tmp] == n
                    UPPER_f1split[t2] = 1e20
                else        
                    UPPER_f1split[t2] = (X_sort[llss_leftstart[t2][where_upper][idx_tmp],f1min]+X_sort[llss_leftstart[t2][where_upper][idx_tmp]+1,f1min])/2
                end
                fs_left = AL_left[where_upper,3]
                fs_right = AL_left[where_upper,4]
                if exh_f1[t2][where_upper] == 0
                    UPPER_LL[t2] = fs_left[argmin(llss_leftleft[t2][where_upper][where_minleft[where_upper]+1,:])]
                    UPPER_LR[t2] = fs_right[argmin(llss_leftright[t2][where_upper][where_minleft[where_upper],:])]
                else
                    UPPER_LL[t2] = fs_left[argmin(llss_leftleft[t2][where_upper][where_minleft[where_upper],:])]
                    UPPER_LR[t2] = fs_right[argmin(llss_leftright[t2][where_upper][where_minleft[where_upper],:])]
                end
            end

            minimum_right = zeros(n_rows_right)
            where_minright = zeros(Int, n_rows_right)
            for i_right = 1:n_rows_right
                if exh_f2[t2][i_right] == 0
                    where_minright[i_right] = argmin(llss_rightleft2[t2][i_right][2:end] .+ llss_rightright2[t2][i_right][1:end-1])
                    minimum_right[i_right] = minimum(llss_rightleft2[t2][i_right][2:end] .+ llss_rightright2[t2][i_right][1:end-1])
                else
                    where_minright[i_right] = argmin(llss_rightleft2[t2][i_right] .+ llss_rightright2[t2][i_right])
                    minimum_right[i_right] = minimum(llss_rightleft2[t2][i_right] .+ llss_rightright2[t2][i_right])
                end
            end
            if minimum(minimum_right) < UPPER_right[t2]
                UPPER_right[t2] = minimum(minimum_right)
                where_upper = argmin(minimum_right)
                f2min = AL_right[where_upper,1]
                UPPER_f2min[t2] = f2min
                idx_tmp = 0
                if exh_f2[t2][where_upper] == 0
                    idx_tmp = where_minright[where_upper] + 1
                else
                    idx_tmp = where_minright[where_upper] 
                end
                if llss_rightstart[t2][where_upper][idx_tmp] == 0
                    UPPER_f2split[t2] = -1e20
                elseif llss_rightstart[t2][where_upper][idx_tmp] == n
                    UPPER_f2split[t2] = 1e20
                else        
                    UPPER_f2split[t2] = (X_sort[llss_rightstart[t2][where_upper][idx_tmp],f2min]+X_sort[llss_rightstart[t2][where_upper][idx_tmp]+1,f2min])/2
                end
                fs_left = AL_right[where_upper,3]
                fs_right = AL_right[where_upper,4]
                if exh_f2[t2][where_upper] == 0
                    UPPER_RL[t2] = fs_left[argmin(llss_rightleft[t2][where_upper][where_minright[where_upper]+1,:])]
                    UPPER_RR[t2] = fs_right[argmin(llss_rightright[t2][where_upper][where_minright[where_upper],:])]
                else
                    UPPER_RL[t2] = fs_left[argmin(llss_rightleft[t2][where_upper][where_minright[where_upper],:])]
                    UPPER_RR[t2] = fs_right[argmin(llss_rightright[t2][where_upper][where_minright[where_upper],:])]
                end
            end
            if time() - st > timelimit
                #print("OK1")
                #sleep(0.3)
                break
            end   
        end
        if time() - st > timelimit
            #print("OK2")
            #sleep(0.3)
            break
        end   
        
        ### update upper bound
        if exh_f0 == 0
            if minimum(UPPER_left[2:end] .+ UPPER_right[1:end-1]) < ub
                find_new_tree = 1
                ub = minimum(UPPER_left[2:end] .+ UPPER_right[1:end-1])
                wmin = argmin(UPPER_left[2:end] .+ UPPER_right[1:end-1])
                best_tree_label[1] = f0
                best_tree_label[2] = partend_f0[wmin+1]
                best_tree_label[3] = [UPPER_f1min[wmin+1], UPPER_f1split[wmin+1], UPPER_LL[wmin+1], UPPER_LR[wmin+1]]
                best_tree_label[4] = [UPPER_f2min[wmin], UPPER_f2split[wmin], UPPER_RL[wmin], UPPER_RR[wmin]]
            end
        else
            if minimum(UPPER_left .+ UPPER_right) < ub
                find_new_tree = 1
                ub = minimum(UPPER_left .+ UPPER_right)
                wmin = argmin(UPPER_left .+ UPPER_right)
                best_tree_label[1] = f0
                best_tree_label[2] = partend_f0[wmin]
                best_tree_label[3] = [UPPER_f1min[wmin], UPPER_f1split[wmin], UPPER_LL[wmin], UPPER_LR[wmin]]
                best_tree_label[4] = [UPPER_f2min[wmin], UPPER_f2split[wmin], UPPER_RL[wmin], UPPER_RR[wmin]]
            end
        end

        for t2 = 1:counttotal_f0
            idx_leftall = order_f0[1:partstart_f0[t2]-1]
            idx_rightall = order_f0[partend_f0[t2]+1:n]
            AL_leftnew = Array{Any,2}(undef, 0, 5)
            n_rows_left = size(AL_left)[1]

    
            for i_left = 1:n_rows_left
                f1 = AL_left[i_left, 1]
                X_leftf1 = X[idx_leftall,f1]
                fs_left = AL_left[i_left,3]
                fs_right = AL_left[i_left,4] 
                counttotal_f1 = length(llss_leftleft2[t2][i_left])
                for t2_left = 1:counttotal_f1
                    
                    LS_RIGHTLEFT2, LS_RIGHTRIGHT2 = 1e20, 1e20
                    if exh_f0 == 0
                        if t2 < counttotal_f0
                            LS_RIGHTRIGHT2 = UPPER_left[t2+1]
                            LS_RIGHTLEFT2 = UPPER_left[t2+1]
                        else
                            LS_RIGHTRIGHT2 = 1e20
                            LS_RIGHTLEFT2 = 1e20
                        end
                    else
                        LS_RIGHTRIGHT2 = UPPER_left[t2]
                        LS_RIGHTLEFT2 = UPPER_left[t2]
                    end
                    
                    LS_RIGHT = minimum(llss_mid[t2, i_left, :] + llss_right[t2])
                    fs_left_new = findall(x -> x < ub - LS_RIGHT + 1e-8, llss_leftleft3[t2][i_left][t2_left,:])
                    fs_left_new2 = findall(x -> x < LS_RIGHTRIGHT2 + 1e-8, llss_leftleft3[t2][i_left][t2_left,fs_left_new])
                    
                    fs_right_new = findall(x -> x < ub - LS_RIGHT + 1e-8, llss_leftright3[t2][i_left][t2_left,:])
                    fs_right_new2 = findall(x -> x < LS_RIGHTLEFT2 + 1e-8, llss_leftright3[t2][i_left][t2_left,fs_right_new])
                    
                    if length(fs_left_new2) > 0 && length(fs_right_new2) > 0
                        
                        AL_leftnew_row = Array{Any}(undef, 1, 5)
                        AL_leftnew_row[1, 1] = AL_left[i_left][1]
                        if exh_f1[t2][i_left] == 0
                            if llss_leftstart[t2][i_left][t2_left] == 0
                                threshold1 = -1e20
                            elseif llss_leftstart[t2][i_left][t2_left] == n
                                threshold1 = 1e20
                            else        
                                threshold1 = (X_sort[llss_leftstart[t2][i_left][t2_left],f1]+X_sort[llss_leftstart[t2][i_left][t2_left]+1,f1])/2
                            end
                            if llss_leftstart[t2][i_left][t2_left+1] == 0
                                threshold2 = -1e20
                            elseif llss_leftstart[t2][i_left][t2_left+1] == n
                                threshold2 = 1e20
                            else        
                                threshold2 = (X_sort[llss_leftstart[t2][i_left][t2_left+1],f1]+X_sort[llss_leftstart[t2][i_left][t2_left+1]+1,f1])/2
                            end
                            if sum((X_leftf1 .>= threshold1) .& (X_leftf1 .<= threshold2)) >= 1
                                AL_leftnew_row[1, 2] = [llss_leftstart[t2][i_left][t2_left]+1, llss_leftstart[t2][i_left][t2_left+1]]
                            else
                                continue
                            end
                            AL_leftnew_row[1, 2] = [llss_leftstart[t2][i_left][t2_left]+1, llss_leftstart[t2][i_left][t2_left+1]]
                        else
                            AL_leftnew_row[1, 2] = [llss_leftstart[t2][i_left][t2_left]+1, llss_leftstart[t2][i_left][t2_left]]
                        end
                        AL_leftnew_row[1, 3] = fs_left[fs_left_new[fs_left_new2]]
                        AL_leftnew_row[1, 4] = fs_right[fs_right_new[fs_right_new2]]
                        if exh_f1[t2][i_left] == 1
                            AL_leftnew_row[1, 5] = minimum(llss_leftleft[t2][i_left][t2_left,:]) + minimum(llss_leftright[t2][i_left][t2_left,:])
                        else
                            AL_leftnew_row[1, 5] = 1e20
                        end
                        AL_leftnew = vcat(AL_leftnew, AL_leftnew_row)
                    end
                    
                end
                if counttotal_f1 == 0 
                    AL_leftnew_row = deepcopy(AL_left[i_left:i_left,:])
                    AL_leftnew = vcat(AL_leftnew, AL_leftnew_row)
                end
            end

            AL_rightnew = Array{Any,2}(undef, 0, 5)
            n_rows_right = size(AL_right)[1]
            for i_right = 1:n_rows_right
                f2 = AL_right[i_right, 1]
                X_rightf2 = X[idx_rightall,f2]
                fs_left = AL_right[i_right,3]
                fs_right = AL_right[i_right,4] 
                counttotal_f2 = length(llss_rightleft2[t2][i_right])
                for t2_right = 1:counttotal_f2

                    LS_LEFTLEFT2, LS_LEFTRIGHT2 = 1e20, 1e20
                    if exh_f0 == 0
                        if t2 > 1 
                            LS_LEFTRIGHT2 = UPPER_right[t2-1]
                            LS_LEFTLEFT2 = UPPER_right[t2-1]
                        else
                            LS_LEFTRIGHT2 = 1e20
                            LS_LEFTLEFT2 = 1e20
                        end
                    else
                        LS_LEFTRIGHT2 = UPPER_right[t2]
                        LS_LEFTLEFT2 = UPPER_right[t2]
                    end
                    LS_LEFT = minimum(llss_mid[t2, :, i_right] + llss_left[t2])
                    fs_left_new = findall(x -> x < ub - LS_LEFT + 1e-8, llss_rightleft3[t2][i_right][t2_right,:])
                    fs_left_new2 = findall(x -> x < LS_LEFTRIGHT2 + 1e-8, llss_rightleft3[t2][i_right][t2_right,fs_left_new])
                    fs_right_new = findall(x -> x < ub - LS_LEFT + 1e-8, llss_rightright3[t2][i_right][t2_right,:])
                    fs_right_new2 = findall(x -> x < LS_LEFTLEFT2 + 1e-8, llss_rightright3[t2][i_right][t2_right,fs_right_new])

                    if length(fs_left_new2) > 0 && length(fs_right_new2) > 0
                      
                        AL_rightnew_row = Array{Any}(undef, 1, 5)
                        AL_rightnew_row[1, 1] = AL_right[i_right][1]
                        if exh_f2[t2][i_right] == 0
                            if llss_rightstart[t2][i_right][t2_right] == 0
                                threshold1 = -1e20
                            elseif llss_rightstart[t2][i_right][t2_right] == n
                                threshold1 = 1e20
                            else        
                                threshold1 = (X_sort[llss_rightstart[t2][i_right][t2_right],f2]+X_sort[llss_rightstart[t2][i_right][t2_right]+1,f2])/2
                            end
                            if llss_rightstart[t2][i_right][t2_right+1] == 0
                                threshold2 = -1e20
                            elseif llss_rightstart[t2][i_right][t2_right+1] == n
                                threshold2 = 1e20
                            else        
                                threshold2 = (X_sort[llss_rightstart[t2][i_right][t2_right+1],f2]+X_sort[llss_rightstart[t2][i_right][t2_right+1]+1,f2])/2
                            end
                            if sum((X_rightf2 .>= threshold1) .& (X_rightf2 .<= threshold2)) >= 1
                                AL_rightnew_row[1, 2] = [llss_rightstart[t2][i_right][t2_right]+1, llss_rightstart[t2][i_right][t2_right+1]]
                            else
                                continue
                            end
                            AL_rightnew_row[1, 2] = [llss_rightstart[t2][i_right][t2_right]+1, llss_rightstart[t2][i_right][t2_right+1]]
                        else
                            AL_rightnew_row[1, 2] = [llss_rightstart[t2][i_right][t2_right]+1, llss_rightstart[t2][i_right][t2_right]]
                        end
                        AL_rightnew_row[1, 3] = fs_left[fs_left_new[fs_left_new2]]
                        AL_rightnew_row[1, 4] = fs_right[fs_right_new[fs_right_new2]]
                        if exh_f2[t2][i_right] == 1
                            AL_rightnew_row[1, 5] = minimum(llss_rightleft[t2][i_right][t2_right,:]) + minimum(llss_rightright[t2][i_right][t2_right,:])
                        else
                            AL_rightnew_row[1, 5] = 1e20
                        end
                        AL_rightnew = vcat(AL_rightnew, AL_rightnew_row)
                    end
                end
                if counttotal_f2 == 0
                    AL_rightnew_row = deepcopy(AL_right[i_right:i_right,:])
                    AL_rightnew = vcat(AL_rightnew, AL_rightnew_row)
                end
            end

            if size(AL_leftnew)[1] == 0 || size(AL_rightnew)[1] == 0
                continue
            end
            AALL_new_row = Array{Any}(undef, 1,4)
            AALL_new_row[1, 1] = AALL[i][1]
            AALL_new_row[1, 2] = [partend_f0[t2]+1,partstart_f0[t2]-1]
            AALL_new_row[1, 3] = AL_leftnew
            AALL_new_row[1, 4] = AL_rightnew
            if i_root == 0
                AALL_new = AALL_new_row
                i_root = 1
            else
                AALL_new = vcat(AALL_new, AALL_new_row)
            end
            if time() - st > timelimit
                #print("OK3")
                #sleep(0.3)
                break
            end   
        end
        if time() - st > timelimit
            #print("OK4")
            #sleep(0.3)
            break
        end   
    end

    ### update AALL_new
    n_rowsnew = size(AALL_new)[1]
    list_new = Vector{Int64}()
    for i = 1:n_rowsnew
        if i == n_rowsnew || AALL_new[i,:] != AALL_new[i+1,:] 
            append!(list_new, i)
        end
    end
    AALL_new = AALL_new[list_new,:]

    n_rowsnew = size(AALL_new)[1]
    for i = 1:n_rowsnew   
        range = AALL_new[i,2]
        AL_left = AALL_new[i,3]
        AL_right = AALL_new[i,4]
        n_rows_left = size(AL_left)[1]
        n_rows_right = size(AL_right)[1]
        if range[2] < range[1]
            leftlist_new = Vector{Int64}()
            minimum_left = minimum(AL_left[:,5])
            for i_left = 1:n_rows_left
                range_left = AL_left[i_left,2]
                if range_left[2] >= range_left[1] || minimum_left >= AL_left[i_left,5]
                    if i_left == n_rows_left || AL_left[i_left,:] != AL_left[i_left+1,:]
                        append!(leftlist_new, i_left)
                    end
                end
            end

            rightlist_new = Vector{Int64}()
            minimum_right = minimum(AL_right[:,5])
            for i_right = 1:n_rows_right
                range_right = AL_right[i_right,2]
                if range_right[2] >= range_right[1] || minimum_right >= AL_right[i_right,5]
                    if i_right == n_rows_right || AL_right[i_right,:] != AL_right[i_right+1,:]
                        append!(rightlist_new, i_right)
                    end
                end
            end
            AALL_new[i,3] = AL_left[leftlist_new,:]
            AALL_new[i,4] = AL_right[rightlist_new,:]
        end
    end

    ### update best tree
    if find_new_tree == 1
        f0, split0 = Int(best_tree_label[1]), best_tree_label[2]
        f1, split1 = Int(best_tree_label[3][1]), best_tree_label[3][2]
        f2, split2 = Int(best_tree_label[4][1]), best_tree_label[4][2]
        f11, f12 = Int(best_tree_label[3][3]), Int(best_tree_label[3][4])
        f21, f22 = Int(best_tree_label[4][3]), Int(best_tree_label[4][4])
        x_ordered = X_sort[:,f0]
        order_f0 = X_sortperm[:,f0]
        idx_left = order_f0[1:split0]
        idx_LL = idx_left[findall(x -> x < split1, X[idx_left, f1])]
        idx_LR =  idx_left[findall(x -> x >= split1, X[idx_left, f1])]
        idx_right = order_f0[split0+1:n]
        idx_RL = idx_right[findall(x -> x < split2, X[idx_right, f2])]
        idx_RR =  idx_right[findall(x -> x >= split2, X[idx_right, f2])]
        l11, b11, lm11, rm11 = one_pass_search_mr(X, y, idx_LL, f11, treetype)
        l12, b12, lm12, rm12 = one_pass_search_mr(X, y, idx_LR, f12, treetype)
        l21, b21, lm21, rm21 = one_pass_search_mr(X, y, idx_RL, f21, treetype)
        l22, b22, lm22, rm22 = one_pass_search_mr(X, y, idx_RR, f22, treetype)
        if abs(ub - (l11 + l12 + l21 + l22)) > 1e-8
            println(ub)
            println(l11 + l12 + l21 + l22)
            sleep(0.1)
            throw(ErrorException)
        end
        best_tree = Array{Any,1}(undef, 4)
        best_tree[1] = best_tree_label[1]
        if split0 == 0
            best_tree[2] = -1e20
        elseif split0 >= length(x_ordered)
            best_tree[2] = 1e20
        else 
            best_tree[2] = 0.5*(x_ordered[split0] + x_ordered[split0+1])
        end
        best_tree[3] = Array{Any,1}(undef, 4)
        best_tree[3][1], best_tree[3][2] = f1, split1
        best_tree[4] = Array{Any,1}(undef, 4)
        best_tree[4][1], best_tree[4][2] = f2, split2
        best_tree[3][3] = Array{Any,1}(undef, 4)
        best_tree[3][4] = Array{Any,1}(undef, 4)
        best_tree[4][3] = Array{Any,1}(undef, 4)
        best_tree[4][4] = Array{Any,1}(undef, 4)
        best_tree[3][3][1], best_tree[3][3][2], best_tree[3][3][3], best_tree[3][3][4] = f11, b11, lm11, rm11
        best_tree[3][4][1], best_tree[3][4][2], best_tree[3][4][3], best_tree[3][4][4] = f12, b12, lm12, rm12
        best_tree[4][3][1], best_tree[4][3][2], best_tree[4][3][3], best_tree[4][3][4] = f21, b21, lm21, rm21
        best_tree[4][4][1], best_tree[4][4][2], best_tree[4][4][3], best_tree[4][4][4] = f22, b22, lm22, rm22

        #ub, best_tree = Localsearch(X, y, 3, best_tree, 10)
    end

    return AALL_new, ub, best_tree
    
   

    
end