using LinearAlgebra
include("gen_data.jl")
include("Algorithms.jl")

function lower_bound_mid(X, y, f0, range, fs_left, fs_right, s, method, treetype = "R")
    
    nn = range[2] - range[1] + 1
    if nn <= 2
        return zeros(n_left, n_right)    
    end 
    n_left = size(fs_left)[1]
    n_right = size(fs_right)[1]
    
    if method == 0
        return zeros(n_left, n_right)    

    elseif method == 1 ## exhaustive search
        
        lb_mid_left = zeros(nn-1, n_left) 
        lb_mid_right = zeros(nn-1, n_right)
        order_f0 = X_sortperm[:,f0]
        for t = 1: nn-1
            for j1 = 1:n_left
                if t >= 2
                    lb_mid_left[t,j1], _, _, _ = one_pass_search_mr(X, y, order_f0[range[1]:range[1]+t], fs_left[j1], treetype)
                end
            end
            
            for j2 = 1:n_right
                if t <= nn-2
                    lb_mid_right[t,j2], _, _, _ = one_pass_search_mr(X, y, order_f0[range[1]+t+1:range[2]], fs_right[j2], treetype)
                end
            end         
        end
        
        LB_MID = zeros(n_left, n_right)
        for j1 = 1:n_left
            for j2 = 1:n_right
                LB_MID[j1, j2] = minimum(lb_mid_left[:, j1] .+ lb_mid_right[:, j2])
            end
        end
        return LB_MID

    elseif method == 2
        
        ## skipping search, drop middle part
        quantiles = [i/s for i=0:s]
        new_pos = convert(Array{Int64,1}, round.(quantiles.*nn))
        
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

        if unique_f0 <= s+1
            lb_mid_left = zeros(unique_f0+1, n_left)
            lb_mid_right = zeros(unique_f0+1, n_right)
            for t2 = 1:unique_f0+1
                for j1 = 1:n_left
                    lb_mid_left[t2, j1], _, _, _ = one_pass_search_mr(X, y, order_f0[range[1]:part_f0[t2]], fs_left[j1], treetype)
                end
                for j2 = 1:n_right
                    lb_mid_right[t2, j2], _, _, _ = one_pass_search_mr(X, y, order_f0[part_f0[t2]+1:range[2]], fs_right[j2], treetype)
                end
            end
        else
            lb_mid_left = zeros(count_f0, n_left)
            lb_mid_right = zeros(count_f0, n_right)
            for t2 = 1:count_f0
                for j1 = 1:n_left
                    lb_mid_left[t2, j1], _, _, _ = one_pass_search_mr(X, y, order_f0[range[1]:spart_f0[t2]], fs_left[j1], treetype)
                end
                for j2 = 1:n_right
                    lb_mid_right[t2, j2], _, _, _ = one_pass_search_mr(X, y, order_f0[spart_f0[t2+1]+1:range[2]], fs_right[j2], treetype)
                end
            end
        end
        
        LB_MID = zeros(n_left, n_right)
        for j1 = 1:n_left
            for j2 = 1:n_right
                LB_MID[j1, j2] = minimum(lb_mid_left[:, j1] .+ lb_mid_right[:, j2])
            end
        end

        return LB_MID
    end    
end

function lower_bound_mid_3D(X, y, f0, range, AL_left, AL_right, s, mid_method, mid_ratio, treetype)
    
    n, _ = size(X)
    nn = range[2] - range[1] + 1
    n_rows_left = size(AL_left)[1]
    n_rows_right = size(AL_right)[1]

    if nn <= 4 || mid_method == 0 || s <= 1
        return zeros(n_rows_left, n_rows_right)
    end

    
    quantiles = [i/s for i=0:s]
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



    llss_left = zeros(counttotal_f0, n_rows_left)
    llss_leftleft = Array{Any, 1}(undef, counttotal_f0)
    llss_leftright = Array{Any, 1}(undef, counttotal_f0)
    llss_leftstart = Array{Any, 1}(undef, counttotal_f0)
    LB_leftMID = Array{Any, 1}(undef, counttotal_f0)
    exh_f1 = Array{Any, 1}(undef, counttotal_f0)
    issingle_f1 = Array{Any, 1}(undef, counttotal_f0)

    llss_right = zeros(counttotal_f0, n_rows_right)
    llss_rightleft = Array{Any, 1}(undef, counttotal_f0)
    llss_rightright = Array{Any, 1}(undef, counttotal_f0)
    llss_rightstart = Array{Any, 1}(undef, counttotal_f0)
    LB_rightMID = Array{Any, 1}(undef, counttotal_f0)
    exh_f2 = Array{Any, 1}(undef, counttotal_f0)
    issingle_f2 = Array{Any, 1}(undef, counttotal_f0)


    for t2 = 1:counttotal_f0

        # left node
        idx_left = order_f0[range[1]:partend_f0[t2]]
        n_left = length(idx_left)
        
        llss_leftleft[t2] = Array{Any, 1}(undef, n_rows_left)
        llss_leftright[t2] = Array{Any, 1}(undef, n_rows_left)
        LB_leftMID[t2] = Array{Any, 1}(undef, n_rows_left)
        llss_leftstart[t2] = Array{Any, 1}(undef, n_rows_left)
        exh_f1[t2] = zeros(n_rows_left)
        issingle_f1[t2] = zeros(n_rows_left)

        for i_left = 1:n_rows_left
            f1 = AL_left[i_left,1]
            range_left = AL_left[i_left,2]
            fs_left = AL_left[i_left,3]
            fs_right = AL_left[i_left,4] 
            num_fs_left = size(fs_left)[1]
            num_fs_right = size(fs_right)[1]  
            
            X_left = X[idx_left, f1]
            od = sortperm(X_left)
            x_ordered = X_left[od] 
            od_left = idx_left[od] 

            count_s1 = X_count[range_left[1],f1] 
            count_e1 = X_count[range_left[2],f1] + 1
            unique_f1 = (count_e1 - count_s1)
            part_f1 = X_invcount[f1][count_s1:count_e1]
            new_pos = count_s1 .+ convert(Array{Int64,1}, round.(quantiles .* unique_f1))
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
            elseif unique_f1 <= s + 1
                exh_f1[t2][i_left] = 1
                issingle_f1[t2][i_left] = 0
                if part_f1[1] == 0
                    part_f1 = part_f1[2:end]
                end
                if part_f1[end] == n
                    part_f1 = part_f1[1:end-1]
                end
                llss_leftstart[t2][i_left] = part_f1
                count_f1 = length(part_f1) - 1
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
                part_total = Vector{Int64}()
                llss_leftstart[t2][i_left] = spart_f1
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
            LB_leftMID[t2][i_left] = zeros(counttotal_f1, num_fs_left, num_fs_right)



            for t2_left = 1:counttotal_f1
                for j1 = 1:num_fs_left 
                    llss_leftleft[t2][i_left][t2_left, j1], _, _, _ = one_pass_search_mr(X, y, od_left[1:partend_f1[t2_left]], fs_left[j1], treetype)
                end
                for j2 = 1:num_fs_right
                    llss_leftright[t2][i_left][t2_left, j2], _, _, _ = one_pass_search_mr(X, y, od_left[partstart_f1[t2_left]:end], fs_right[j2], treetype)
                end
                total_compute = n_left * (counttotal_f1 - 1)
                n_sub = partstart_f1[t2_left] - partend_f1[t2_left] 
                s_mid =  Int(ceil(mid_ratio * total_compute / n_sub))
                if exh_f1[t2][i_left] == 0
                    LB_leftMID[t2][i_left][t2_left,:,:] = lower_bound_mid2(X, y, f1, od_left[partend_f1[t2_left]+1:partstart_f1[t2_left]-1],
                                [llss_leftstart[t2][i_left][t2_left]+1,llss_leftstart[t2][i_left][t2_left+1]], fs_left, fs_right, s_mid, mid_method, treetype)
                else
                    LB_leftMID[t2][i_left][t2_left,:,:] = zeros(num_fs_left, num_fs_right)
                end
            end
            if counttotal_f1 > 0
                llss_tmp = zeros(num_fs_left, num_fs_right)
                for j1 = 1:num_fs_left
                    for j2 = 1:num_fs_right
                        llss_tmp[j1,j2] = minimum(LB_leftMID[t2][i_left][:,j1,j2]+llss_leftleft[t2][i_left][:,j1]+llss_leftright[t2][i_left][:,j2])
                    end
                end
                llss_left[t2,i_left] = minimum(llss_tmp)
            else
                llss_left[t2,i_left] = 0
            end
        end


        # right node
        idx_right = order_f0[partstart_f0[t2]:range[2]]
        n_right = length(idx_right)

        llss_rightleft[t2] = Array{Any, 1}(undef, n_rows_right)
        llss_rightright[t2] = Array{Any, 1}(undef, n_rows_right)
        LB_rightMID[t2] = Array{Any, 1}(undef, n_rows_right)
        llss_rightstart[t2] = Array{Any, 1}(undef, n_rows_right)
        exh_f2[t2] = zeros(n_rows_right)
        issingle_f2[t2] = zeros(n_rows_right)

        for i_right = 1:n_rows_right
            f2 = AL_right[i_right,1]
            range_right = AL_right[i_right,2]
            fs_left = AL_right[i_right,3]
            fs_right = AL_right[i_right,4] 
            num_fs_left = size(fs_left)[1]
            num_fs_right = size(fs_right)[1]  
            
            X_right = X[idx_right, f2]
            od = sortperm(X_right)
            x_ordered = X_right[od] 
            od_right = idx_right[od] 

            count_s2 = X_count[range_right[1],f2] 
            count_e2 = X_count[range_right[2],f2] + 1
            unique_f2 = (count_e2 - count_s2)
            part_f2 = X_invcount[f2][count_s2:count_e2]
            new_pos = count_s2 .+ convert(Array{Int64,1}, round.(quantiles .* unique_f2))
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
            elseif unique_f2 <= s + 1
                exh_f2[t2][i_right] = 1
                issingle_f2[t2][i_right] = 0
                llss_rightstart[t2][i_right] = part_f2
                if part_f2[1] == 0
                    part_f2 = part_f2[2:end]
                end
                if part_f2[end] == n
                    part_f2 = part_f2[1:end-1]
                end
                count_f2 = length(part_f2) - 1
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
                llss_rightstart[t2][i_right] = spart_f2
                part_total = Vector{Int64}()
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
            LB_rightMID[t2][i_right] = zeros(counttotal_f2, num_fs_left, num_fs_right)

            

            for t2_right = 1:counttotal_f2
                for j1 = 1:num_fs_left
                    llss_rightleft[t2][i_right][t2_right, j1], _, _, _ = one_pass_search_mr(X, y, od_right[1:partend_f2[t2_right]], fs_left[j1], treetype)
                end
                for j2 = 1:num_fs_right
                    llss_rightright[t2][i_right][t2_right, j2], _, _, _ = one_pass_search_mr(X, y, od_right[partstart_f2[t2_right]:end], fs_right[j2], treetype)
                end
                total_compute = n_right * (counttotal_f2 - 1)
                n_sub = partstart_f2[t2_right] - partend_f2[t2_right] 
                s_mid =  Int(ceil(mid_ratio * total_compute / n_sub))
                if exh_f2[t2][i_right] == 0
                    LB_rightMID[t2][i_right][t2_right,:,:] = lower_bound_mid2(X, y, f2, od_right[partend_f2[t2_right]+1:partstart_f2[t2_right]-1],
                                [llss_rightstart[t2][i_right][t2_right]+1,llss_rightstart[t2][i_right][t2_right+1]], fs_left, fs_right, s_mid, mid_method, treetype)
                else
                    LB_rightMID[t2][i_right][t2_right,:,:] = zeros(num_fs_left, num_fs_right)
                end
            end
            if counttotal_f2 > 0
                llss_tmp = zeros(num_fs_left, num_fs_right)
                for j1 = 1:num_fs_left
                    for j2 = 1:num_fs_right
                        llss_tmp[j1,j2] = minimum(LB_rightMID[t2][i_right][:,j1,j2]+llss_rightleft[t2][i_right][:,j1]+llss_rightright[t2][i_right][:,j2])
                    end
                end
                llss_right[t2,i_right] = minimum(llss_tmp)
            else
                llss_right[t2,i_right] = 0
            end
        end
    end
    llss = zeros(n_rows_left, n_rows_right)
    for i_left = 1:n_rows_left
        for i_right = 1:n_rows_right
            llss[i_left, i_right] = minimum(llss_right[:,i_right] + llss_left[:,i_left])
        end
    end
    return llss
end

function lower_bound_mid2(X, y, f0, idx, range, fs_left, fs_right, s, method, treetype)
    
    nn = length(idx)
    n, _ = size(X)
    n_left = size(fs_left)[1]
    n_right = size(fs_right)[1]

    if nn <= 2
        return zeros(n_left, n_right)    
    end 
    if method == 0
        return zeros(n_left, n_right)    

    elseif method == 1 ## exhaustive search
        
        lb_mid_left = zeros(nn-1, n_left) 
        lb_mid_right = zeros(nn-1, n_right)
        for t = 1: nn-1
            for j1 = 1:n_left
                if t >= 2
                    lb_mid_left[t,j1], _, _, _ = one_pass_search_mr(X, y, idx[1:t], fs_left[j1], treetype)
                end
            end
            
            for j2 = 1:n_right
                if t <= nn-2
                    lb_mid_right[t,j2], _, _, _ = one_pass_search_mr(X, y, idx[t+1:end], fs_right[j2], treetype)
                end
            end         
        end
        
        LB_MID = zeros(n_left, n_right)
        for j1 = 1:n_left
            for j2 = 1:n_right
                LB_MID[j1, j2] = minimum(lb_mid_left[:, j1] .+ lb_mid_right[:, j2])
            end
        end
        return LB_MID

    elseif method == 2

        
        ## skipping search, drop middle part
        quantiles = [i/s for i=0:s]
        
        num_fs_left = size(fs_left)[1]
        num_fs_right = size(fs_right)[1]  
        
        if nn < n / 10
            X_mid = X[idx, f0]
            y_mid = y[idx,:]
            od = sortperm(X_mid)
            x_ordered = X_mid[od] 
            y_ordered = y_mid[od,:]
            od_mid = idx[od] 
        else
            od = zeros(Int, nn+1)
            tmp = zeros(Int, n)
            tmp[X_invperm[idx,f0]] = idx
            count = 1
            @inbounds for isort=1:n
                od[count] = tmp[isort]
                count += (tmp[isort] != zero(Int))
            end
            od_mid = od[1:end-1]
            x_ordered = X[od_mid,f0]
            y_ordered = y[od_mid,:]
        end

        count_s0 = X_count[range[1],f0] 
        count_e0 = X_count[range[2],f0] + 1
        unique_f0 = (count_e0 - count_s0)
        part_f0 = X_invcount[f0][count_s0:count_e0]
        new_pos = count_s0 .+ convert(Array{Int64,1}, round.(quantiles .* unique_f0))
        spart_f0 = X_invcount[f0][new_pos]
        count_f0 = length(spart_f0) - 1 
        partend_f0, partstart_f0, counttotal_f0 = 0, 0, 0
        if unique_f0 <= s + 1
            if part_f0[1] == 0
                part_f0 = part_f0[2:end]
                unique_f0 -= 1
            end
            if part_f0[end] == n
                part_f0 = part_f0[1:end-1]
                unique_f0 -= 1
            end
            count_f0 = length(part_f0) - 1
            part_total = Vector{Int64}()
            count_tmp = 1
            if part_f0[count_tmp] == 0
                threshold = -1e20
            elseif part_f0[count_tmp] == n
                threshold = 1e20
            else        
                threshold = (X_sort[part_f0[count_tmp],f0]+X_sort[part_f0[count_tmp]+1,f0]) / 2
            end
            for t_mid = 1:nn
                while x_ordered[t_mid] > threshold
                    append!(part_total,t_mid-1)
                    count_tmp += 1
                    if count_tmp <= count_f0 + 1
                        if part_f0[count_tmp] == 0
                            threshold = -1e20
                        elseif part_f0[count_tmp] == n
                            threshold = 1e20
                        else        
                            threshold = (X_sort[part_f0[count_tmp],f0]+X_sort[part_f0[count_tmp]+1,f0]) / 2
                        end
                    else
                        threshold = 1e20
                    end
                end
            end
            while count_tmp <= count_f0 + 1
                append!(part_total,nn)
                count_tmp += 1
            end
            partend_f0 = part_total
            partstart_f0 = part_total .+ 1
            counttotal_f0 = length(part_total) 
        else
            part_total = Vector{Int64}()
            count_tmp = 1
            if spart_f0[count_tmp] == 0
                threshold = -1e20
            elseif spart_f0[count_tmp] == n
                threshold = 1e20
            else        
                threshold = (X_sort[spart_f0[count_tmp],f0]+X_sort[spart_f0[count_tmp]+1,f0]) / 2
            end
            for t_mid = 1:nn
                while x_ordered[t_mid] > threshold
                    append!(part_total,t_mid-1)
                    count_tmp += 1
                    if count_tmp <= count_f0 + 1
                        if spart_f0[count_tmp] == 0
                            threshold = -1e20
                        elseif spart_f0[count_tmp] == n
                            threshold = 1e20
                        else        
                            threshold = (X_sort[spart_f0[count_tmp],f0]+X_sort[spart_f0[count_tmp]+1,f0]) / 2
                        end
                    else
                        threshold = 1e20
                    end
                end
            end
            while count_tmp <= count_f0 + 1
                append!(part_total,nn)
                count_tmp += 1
            end
            counttotal_f0 = length(part_total) - 1
            partend_f0 = part_total[1:counttotal_f0] 
            partstart_f0 = part_total[2:counttotal_f0+1] .+ 1
        end
        lb_mid_left = zeros(counttotal_f0, num_fs_left)
        lb_mid_right = zeros(counttotal_f0, num_fs_right)
        for t2_mid = 1:counttotal_f0
            for j1 = 1:num_fs_left 
                lb_mid_left[t2_mid, j1], _, _, _ = one_pass_search_mr(X, y, od_mid[1:partend_f0[t2_mid]], fs_left[j1], treetype)
            end
            for j2 = 1:num_fs_right
                lb_mid_right[t2_mid, j2], _, _, _ = one_pass_search_mr(X, y, od_mid[partstart_f0[t2_mid]:end], fs_right[j2], treetype)
            end
        end
        
        LB_MID = zeros(n_left, n_right)
        for j1 = 1:n_left
            for j2 = 1:n_right
                LB_MID[j1, j2] = minimum(lb_mid_left[:, j1] .+ lb_mid_right[:, j2])
            end
        end

        return LB_MID
    end    
end