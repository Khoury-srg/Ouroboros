using NeuralVerification, LazySets
import NeuralVerification: get_gradient_bounds, get_gradient

function verify(nnet_file, X_spec, Y_spec; max_iter=100, sampling_size=20, is_polytope=false)
    net = read_nnet(nnet_file)
    # println("=========")
    # println(nnet_file)
    # print(net)
    # println(X_spec)
    # println(Y_spec)
    X = is_polytope ? HPolytope(X_spec[1], X_spec[2]) : Hyperrectangle(low=X_spec[1], high=X_spec[2])
    Y = HPolytope(Y_spec[1], Y_spec[2])
    solver = Ai2z()
    prob = Problem(net, X, Y)
    # println("start solving")
    res = solve(solver, prob, max_iter=max_iter, sampling_size=sampling_size)[1]
    # println("done solving")
    res.status == :violated && (return "violated", res.counter_examples)
    res.status == :unknown && (return "unknown", nothing)

    return "holds", nothing
end

function MonoIncVerify(nnet_file, X_spec, Y_spec; max_iter=1000, sampling_size=20)

    # println("--- in mono verify ---")
    # @show nnet_file
    # @show X_spec
    # @show Y_spec

    dim, direction = Y_spec
    
    dim += 1
    net = read_nnet(nnet_file)
    
    # @show X_spec
    Xs = [Hyperrectangle(low=X_spec[1], high=X_spec[2])]
    function compute_volume(rect)
        volume = 1
        for r in rect.radius
            # println("r = ", r)
            if abs(r) > 0
                volume *= abs(r)
            end
        end
        return volume
    end

    total_volume = compute_volume(Xs[1])

    # return true, 0, nothing
    verified_volume = 0
    counter_examples = []

    is_mono(lb, ub, dim) = direction > 0 ? lb[dim] > 0 : ub[dim] < 0

    # println("total_volume")
    # println(total_volume)

    # println("start iter")
    # println("max_iter ", max_iter)
    for i in 1:max_iter
        X = popfirst!(Xs)
        lb, ub = get_gradient_bounds(net, X)

        # @show is_mono(lb, ub, dim), volume(X)

        if is_mono(lb, ub, dim)
            verified_volume += compute_volume(X)
            isempty(Xs) && break
            continue
        end
        # divide X into two subsets
        X1, X2 = divide(X)
        push!(Xs, X1)
        push!(Xs, X2)
    end
    status = "holds"
    single_sample_size = max(1, sampling_size÷max(1,length(Xs)))
    total_sampled = 0
    for X in Xs
        lb, ub = get_gradient_bounds(net, X)
        # @show dim, direction, lb[dim], ub[dim], is_mono(lb, ub, dim), volume(X)
        if is_mono(lb, ub, dim)
            verified_volume += compute_volume(X)
            continue
        else
            status = "violated"

            if total_sampled > sampling_size
                continue
            end
            sampled = sample_grad_counter_examples(net, X, dim, direction, single_sample_size)
            total_sampled += length(sampled)
            # if length(sampled) == 0
            #     println("empty X:", X)
            # end
            push!(counter_examples, sampled)
        end
    end
    # return "holds", [], 0
    # @show total_volume
    println("verified_volume")
    println(verified_volume)
    println("total_volume")
    println(total_volume)
    println("verified mono ratio: ", verified_volume / total_volume)
    println(length(vcat(counter_examples...)))
    # return "holds", [], 0
    return status, vcat(counter_examples...), verified_volume / total_volume
end

function sample_grad_counter_examples(net, input, dim, direction, sampling_size)
    sampler = LazySets.RejectionSampler(input)
    input_approx = LazySets.box_approximation(input);
    samples = LazySets.sample(input_approx, sampling_size; sampler=sampler)
    counter_examp1es = [x for x in samples if (get_gradient(net, x)[dim]) * direction < 0]
    if length(counter_examp1es) > 5
        @show counter_examp1es[1:5]
        ret = [(x, get_gradient(net, x), dim, direction) for x in counter_examp1es[1:5]]
        @show ret
    end

    return counter_examp1es
end

function prob_verify_Neurify(nnet_file, X_spec, Y_spec, desired_prob; max_iter = 1e9, min_size = 0, sampling_size = 20)
    println("--- in verify ---")
    @show nnet_file
    @show X_spec
    @show Y_spec
    @show desired_prob

    net = read_nnet(nnet_file)
    
    Y = HPolytope(Y_spec[1], Y_spec[2])

    Xs = [Hyperrectangle(low=X_spec[1], high=X_spec[2])]
    

    total_volume = sum([volume(X) for X in Xs])
    @show total_volume
    # return true, 0, nothing
    verified_volume = 0

    counter_examples = []

    remains = []
    # while !isempty(Xs)
    for i = 1:max_iter
        isempty(Xs) && break
        X = popfirst!(Xs)
        
        solver = Neurify(max_iter=1)
        prob = Problem(net, X, Y)
        res = solve(solver, prob, sampling_size=1)[1]
        # res = solve(solver, prob)
        @show res.status, volume(X)
        if res.status == :holds
            verified_volume += volume(X)
            (verified_volume / total_volume > desired_prob) && break
            continue
        end

        if volume(X) < min_size
            push!(remains, X)
        else
            # divide X into two subsets
            X1, X2 = divide(X)
            push!(Xs, X1)
            push!(Xs, X2)
        end
    end

    Xs = [Xs; remains]

    for X in Xs
        solver = Neurify(max_iter=1)
        prob = Problem(net, X, Y)
        res = solve(solver, prob, sampling_size=sampling_size ÷ length(Xs))[1]
        @show res.status, volume(X)
        if res.status == :holds
            verified_volume += volume(X)
            (verified_volume / total_volume > desired_prob) && break
            continue
        end
        res.status == :violated && (push!(counter_examples, res.counter_examples))
    end

    @show total_volume

    println("verified prob: ", verified_volume / total_volume)

    (verified_volume / total_volume > desired_prob) && return "holds", nothing, verified_volume

    return "violated", vcat(counter_examples...), verified_volume
end

function prob_verify_Ai2z(nnet_file, X_spec, Y_spec, desired_prob; max_iter = 1e9, min_size = 0, sampling_size = 20)

    net = read_nnet(nnet_file)
    
    Y = HPolytope(Y_spec[1], Y_spec[2])
    # @show X_spec
    # @show X_spec[1]
    # @show X_spec[2]

    Xs = [Hyperrectangle(low=X_spec[1], high=X_spec[2])]
    println("verifying using Ai2z")

    total_volume = sum([volume(X) for X in Xs])
    # @show total_volume
    # return true, 0, nothing
    verified_volume = 0

    counter_examples = []

    remains = []
    # while !isempty(Xs)
    for i = 1:max_iter
        isempty(Xs) && break
        X = popfirst!(Xs)
        
        solver = Ai2z()
        prob = Problem(net, X, Y)
        res = solve(solver, prob, max_iter=1)[1]
        if res.status === :holds
            verified_volume += volume(X)
            (verified_volume / total_volume > desired_prob) && break
            continue
        end

        if volume(X) < min_size
            push!(remains, X)
        else
            # divide X into two subsets
            X1, X2 = divide(X)
            push!(Xs, X1)
            push!(Xs, X2)
        end
    end

    Xs = [Xs; remains]

    for X in Xs
        solver = Ai2z()
        prob = Problem(net, X, Y)
        res = solve(solver, prob, max_iter=1, sampling_size=sampling_size ÷ length(Xs))[1]
        # @show res.status, volume(X)
        if res.status == :holds
            verified_volume += volume(X)
            (verified_volume / total_volume > desired_prob) && break
            continue
        end
        res.status == :violated && (push!(counter_examples, res.counter_examples))
    end

    println("verified prob for this spec: ", verified_volume / total_volume)

    (verified_volume / total_volume > desired_prob) && return "holds", nothing, verified_volume

    return "violated", vcat(counter_examples...), verified_volume
end


function divide(X)
    high_dim = [high(X, i) for i in 1:dim(X)]
    low_dim = [low(X, i) for i in 1:dim(X)]
    dim_sizes = high_dim - low_dim
    (max_len, max_dim) = findmax(dim_sizes)
    
    l_high_dim, u_low_dim = copy(high_dim), copy(low_dim)

    l_high_dim[max_dim] = low_dim[max_dim] + max_len / 2
    u_low_dim[max_dim] = low_dim[max_dim] + max_len / 2

    if any(l_high_dim < low_dim) || any(high_dim < u_low_dim)
        println("=========================")
        println("error dividing")
        @show low_dim
        @show l_high_dim
        @show u_low_dim
        @show high_dim
    end

    return Hyperrectangle(low=low_dim, high=l_high_dim), Hyperrectangle(low=u_low_dim, high=high_dim)

end


function prob_verify_MIP(nnet_file, X_spec, Y_spec, desired_prob; max_iter=100, sampling_size=20)
    
    net = read_nnet(nnet_file)
    
    Y = HPolytope(Y_spec[1], Y_spec[2])
    Xs = [Hyperrectangle(low=X_spec[1], high=X_spec[2])] # center and radius

    total_volume = sum([volume(X) for X in Xs])
    verified_volume = 0

    counter_examples = []

    for i in 1:max_iter
        X = popfirst!(Xs)
        X1, X2 = divide(X)
        push!(Xs, X1)
        push!(Xs, X2)
    end

    unsafe_Xs = []
    for X in Xs
        solver = MIPVerify()
        prob = Problem(net, X, Y)
        res = solve(solver, prob)
        
        if res.status == :holds
            verified_volume += volume(X)
            (verified_volume / total_volume > desired_prob) && break
            continue
        end
        @show res.max_disturbance, res.max_disturbance ^ dim(X)
        verified_volume += (2*res.max_disturbance) ^ dim(X)
        push!(unsafe_Xs, X)
    end
    println("verified prob: ", verified_volume / total_volume)
    (verified_volume / total_volume > desired_prob) && return true, nothing, verified_volume
    
    counter_examples = vcat([sample_counter_examples(X, Y, net, sampling_size ÷ length(unsafe_Xs)) for X in unsafe_Xs]...)
    return "violated", counter_examples, verified_volume
end

function sample_counter_examples(input, output, nnet, sampling_size)
    sampler = LazySets.RejectionSampler(input)
    input_approx = LazySets.box_approximation(input);
    samples = LazySets.sample(input_approx, sampling_size; sampler=sampler)
    counter_examp1es = [sample for sample in samples if compute_output(nnet, sample) ∉ output]
    return counter_examp1es
end
