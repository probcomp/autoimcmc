# example univariate mixture model from richardson and green
# https://people.maths.bris.ac.uk/~mapjg/papers/RichardsonGreenRSSB.pdf
# implemented using Gen's involutive MCMC construct

using Gen
using PyPlot
import Random

include("dirichlet.jl")
include("mixture_of_normals.jl")

@dist poisson_plus_one(rate) = poisson(rate) + 1

@gen function model(n::Int)
    k ~ poisson_plus_one(1)
    means = [({(:mu, j)} ~ normal(0, 10)) for j in 1:k]
    vars = [({(:var, j)} ~ inv_gamma(1, 10)) for j in 1:k]
    weights ~ dirichlet([2.0 for j in 1:k])
    for i in 1:n
        {(:x, i)} ~ mixture_of_normals(weights, means, vars)
    end
end

function get_n(trace)
    return get_args(trace)[1]
end

function get_means(trace)
    k = trace[:k]
    return [trace[(:mu, i)] for i in 1:k]
end

function get_vars(trace)
    k = trace[:k]
    return [trace[(:var, i)] for i in 1:k]
end

function get_xs(trace)
    n = get_n(trace)
    return [trace[(:x, i)] for i in 1:n]
end

function marginal_density(k, weights, means, vars, x)
    ls = zeros(k)
    for j in 1:k
        mu = means[j]
        var = vars[j]
        ls[j] = logpdf(normal, x, mu, sqrt(var)) + log(weights[j])
    end
    return exp(logsumexp(ls))
end

function get_densities_at(trace, xs)
    k = trace[:k]
    weights = trace[:weights]
    means = get_means(trace)
    vars = get_vars(trace)
    return [marginal_density(k, weights, means, vars, x) for x in xs]
end

function render_trace(trace, xmin, xmax, colors)
    # histogram
    xs = get_xs(trace)
    (hist_data, ) = hist(xs, bins=collect(range(xmin, stop=xmax, length=50)), color="LightGray")
    (ymin, ymax) = gca().get_ylim()

    # density plot
    test_xs = collect(range(xmin, stop=xmax, length=1000))
    densities = get_densities_at(trace, test_xs)
    max_density = maximum(densities)
    scale = ymax / max_density
    plot(test_xs, densities * scale, color="black", linewidth=4, zorder=1)
    
    # individual component density plot
    #if trace[:k] > 1
        for j in 1:trace[:k]
            densities = [marginal_density(1, [1.0], [trace[(:mu, j)]], [trace[(:var, j)]], x) for x in test_xs] * trace[:weights][j]
            max_density = maximum(densities)
            #scale = ymax / max_density
            plot(test_xs, densities * scale, color=colors[j], linewidth=2, zorder=2)
        end
    #end
end

# simulate data and plot a histogram..
function show_prior_data()
    Random.seed!(3)
    n = 100
    trace = simulate(model, (n,))
    figure()
    xmin = -30.0
    xmax = 30.0
    colors = ["red", "orange", "blue"]
    render_trace(trace, xmin, xmax, colors)
    savefig("prior_sample.png")
end

#show_prior_data()

function generate_synthetic_two_mixture_data()
    Random.seed!(1)
    n = 100
    constraints = choicemap()
    constraints[:k] = 3
    constraints[:weights] = [0.4, 0.4, 0.2]
    constraints[(:mu, 1)] = -10.0
    constraints[(:mu, 2)] = 10.0
    constraints[(:mu, 3)] = 20.0
    constraints[(:var, 1)] = 50.0
    constraints[(:var, 2)] = 50.0
    constraints[(:var, 3)] = 50.0
    trace, = generate(model, (n,), constraints)
    #figure()
    #xmin = -30.0
    #xmax = 30.0
    #render_trace(trace, xmin, xmax)
    #savefig("synthetic_data.png")
    return trace
end

#generate_synthetic_two_mixture_data()

function merge_weights(weights, j, k)
    w1 = weights[j]
    w2 = weights[k]
    w = w1 + w2
    u1 = w1 / w
    new_weights = [(i == j) ? w : weights[i] for i in 1:k-1]
    return (new_weights, u1)
end

function merge_mean_and_var(mu1, mu2, var1, var2, w1, w2, w)
    mu = (mu1 * w1 + mu2 * w2) / w
    var = (w1 * (mu1^2 + var1) + w2 * (mu2^2 + var2)) / w - mu^2
    C = (var1 * w1) / (var2 * w2)
    u3 = C / (1 + C)
    u2 = ((mu - mu1) / sqrt(var)) * sqrt(w1 / w2)
    return (mu, var, u2, u3)
end

function split_weights(weights, j, u1, k)
    w = weights[j]
    w1 = w * u1
    w2 = w * (1 - u1)
    new_weights = [(i == j) ? w1 : (i == k + 1) ? w2 : weights[i] for i in 1:k+1]
    @assert isapprox(sum(new_weights), 1.0)
    return new_weights
end

function split_means(mu, var, u2, w1, w2)
    mu1 = mu - u2 * sqrt(var) * sqrt(w2 / w1)
    mu2 = mu + u2 * sqrt(var) * sqrt(w1 / w2)
    return (mu1, mu2)
end

function split_vars(w1, w2, var, u2, u3)
    var1 = u3 * (1 - u2^2) * var * (w1 + w2) / w1
    var2 = (1 - u3) * (1 - u2^2) * var * (w1 + w2) / w2
    return (var1, var2)
end

@gen function split_merge_proposal(trace)
   k = trace[:k]
   split = (k == 1) ? true : ({:split} ~ bernoulli(0.5))
   if split
      # split; pick cluster to split and sample degrees of freedom
      cluster_to_split ~ uniform_discrete(1, k)
      u1 ~ beta(2, 2)
      u2 ~ beta(2, 2)
      u3 ~ beta(1, 1)
   else
      # merge; pick cluster to merge with last cluster
      cluster_to_merge ~ uniform_discrete(1, k-1)
   end
end

@transform split_merge_inv (model_in, aux_in) to (model_out, aux_out) begin
    k = @read(model_in[:k], :discrete)
    split = (k == 1) ? true : @read(aux_in[:split], :discrete)
    if split

        cluster_to_split = @read(aux_in[:cluster_to_split], :discrete)
        u1 = @read(aux_in[:u1], :continuous)
        u2 = @read(aux_in[:u2], :continuous)
        u3 = @read(aux_in[:u3], :continuous)
        weights = @read(model_in[:weights], :continuous)
        mu = @read(model_in[(:mu, cluster_to_split)], :continuous)
        var = @read(model_in[(:var, cluster_to_split)], :continuous)

        new_weights = split_weights(weights, cluster_to_split, u1, k)
        w1 = new_weights[cluster_to_split]
        w2 = new_weights[k+1]
        (mu1, mu2) = split_means(mu, var, u2, w1, w2)
        (var1, var2) = split_vars(w1, w2, var, u2, u3)

        @write(model_out[:k], k+1, :discrete)
        @copy(aux_in[:cluster_to_split], aux_out[:cluster_to_merge])
        @write(aux_out[:split], false, :discrete)
        @write(model_out[:weights], new_weights, :continuous)
        @write(model_out[(:mu, cluster_to_split)], mu1, :continuous)
        @write(model_out[(:mu, k+1)], mu2, :continuous)
        @write(model_out[(:var, cluster_to_split)], var1, :continuous)
        @write(model_out[(:var, k+1)], var2, :continuous)

    else

        cluster_to_merge = @read(aux_in[:cluster_to_merge], :discrete)
        mu1 = @read(model_in[(:mu, cluster_to_merge)], :continuous)
        mu2 = @read(model_in[(:mu, k)], :continuous)
        var1 = @read(model_in[(:var, cluster_to_merge)], :continuous)
        var2 = @read(model_in[(:var, k)], :continuous)
        weights = @read(model_in[:weights], :continuous)
        w1 = weights[cluster_to_merge]
        w2 = weights[k]

        (new_weights, u1) = merge_weights(weights, cluster_to_merge, k)
        w = new_weights[cluster_to_merge]
        (mu, var, u2, u3) = merge_mean_and_var(mu1, mu2, var1, var2, w1, w2, w)
    
        @write(model_out[:k], k-1, :discrete)
        @copy(aux_in[:cluster_to_merge], aux_out[:cluster_to_split])
        if k > 2
            @write(aux_out[:split], true, :discrete)
        end
        @write(model_out[:weights], new_weights, :continuous)
        @write(model_out[(:mu, cluster_to_merge)], mu, :continuous)
        @write(model_out[(:var, cluster_to_merge)], var, :continuous)
        @write(aux_out[:u1], u1, :continuous)
        @write(aux_out[:u2], u2, :continuous)
        @write(aux_out[:u3], u3, :continuous)
    end
end

is_involution!(split_merge_inv)

function split_merge_move(trace)
    return mh(trace, split_merge_proposal, (), split_merge_inv; check=true)
end

# TODO add permutation moves.. (with the last cluster)

function test_split_merge_move()
    Random.seed!(1)
    three_cluster_trace = generate_synthetic_two_mixture_data()
    two_cluster_trace = three_cluster_trace
    num_acc_merge = 0
    for rep in 1:1000
        new_trace, acc = split_merge_move(three_cluster_trace)
        if acc && new_trace[:k] == 2
            num_acc_merge += 1
            println("mu: $(new_trace[(:mu, 1)]), var: $(new_trace[(:var, 1)])")
            two_cluster_trace = new_trace
        end
    end
    println("num_acc_merge: $num_acc_merge")
    @assert num_acc_merge > 0

    one_cluster_trace = two_cluster_trace
    num_acc_merge = 0
    for rep in 1:1000
        new_trace, acc = split_merge_move(two_cluster_trace)
        if acc && new_trace[:k] == 1
            num_acc_merge += 1
            println("mu: $(new_trace[(:mu, 1)]), var: $(new_trace[(:var, 1)])")
            one_cluster_trace = new_trace
        end
    end
    println("num_acc_merge: $num_acc_merge")
    @assert num_acc_merge > 0

    figure(figsize=(8, 2))
    xmin = -30.0
    xmax = 30.0
    subplot(1, 3, 1)
    render_trace(three_cluster_trace, xmin, xmax, ["red", "purple", "blue"])
    gca().get_yaxis().set_visible(false)
    subplot(1, 3, 2)
    render_trace(two_cluster_trace, xmin, xmax, ["red", "orange", "blue"])
    gca().get_yaxis().set_visible(false)
    subplot(1, 3, 3)
    render_trace(one_cluster_trace, xmin, xmax, ["red", "orange", "blue"])
    gca().get_yaxis().set_visible(false)
    tight_layout()
    savefig("rjmcmc.png")


end

test_split_merge_move()
