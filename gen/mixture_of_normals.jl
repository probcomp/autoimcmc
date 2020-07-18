struct MixtureOfNormals <: Gen.Distribution{Float64} end

const mixture_of_normals = MixtureOfNormals()

function Gen.logpdf(
        dist::MixtureOfNormals, x::Real,
        weights::Vector{Float64}, means::Vector{Float64}, vars::Vector{Float64})
    ls = Vector{Float64}(undef, length(means))
    for i=1:length(means)
        ls[i] = Gen.logpdf(normal, x, means[i], sqrt(vars[i])) + log(weights[i])
    end
    return logsumexp(ls)
end

function Gen.random(
        dist::MixtureOfNormals,
        weights::Vector{Float64}, means::Vector{Float64}, vars::Vector{Float64})
    i = Gen.categorical(weights)
    return Gen.random(normal, means[i], sqrt(vars[i]))
end

function Gen.logpdf_grad(
        dist::MixtureOfNormals, x::Real,
        weights::Vector{Float64}, means::Vector{Float64}, vars::Vector{Float64})
    return (nothing, nothing, nothing, nothing)
end

(dist::MixtureOfNormals)(weights, means, vars) = Gen.random(dist, weights, means, vars)
Gen.is_discrete(dist::MixtureOfNormals) = false
Gen.has_output_grad(dist::MixtureOfNormals) = false
Gen.has_argument_grads(dist::MixtureOfNormals) = (false, false, false, false)
