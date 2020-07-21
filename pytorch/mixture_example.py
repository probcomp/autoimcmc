from languages import *

#########################################
# polar vs cartesian coordinate example #
#########################################

Bernoulli = torch.distributions.bernoulli.Bernoulli
Categorical = torch.distributions.categorical.Categorical
Normal = torch.distributions.normal.Normal
Uniform = torch.distributions.uniform.Uniform
Beta = torch.distributions.beta.Beta
MixtureSameFamily = torch.distributions.mixture_same_family.MixtureSameFamily

def p(n):
    k = 2 if sample(Bernoulli(0.5), "two_clusters") else 1
    means = torch.tensor([sample(Normal(0, 10), ("mu", j)) for j in range(1, k+1)])
    stdevs = torch.ones(k)
    weights = torch.ones(k)/k
    for i in range(1, n+1):
        # mixture of normals
        sample(MixtureSameFamily(Categorical(weights), Normal(means, stdevs)), ("x", i))
    return None

def q(model_trace):
    if not model_trace["two_clusters"]:
        # sample extra degree of freedom
        sample(Beta(2, 2), "u")
    return None

def f():
    two_clusters = read(model_in["two_clusters"], discrete)
    write(model_out["two_clusters"], 1 - two_clusters, discrete)
    if not two_clusters:
        # one cluster, split
        mu = read(model_in[("mu", 1)], continuous)
        u = read(aux_in["u"], continuous)
        mu1 = mu - u
        mu2 = mu + u
        write(model_out[("mu", 1)], mu1, continuous)
        write(model_out[("mu", 2)], mu2, continuous)
    else:
        # two clusters, merge
        mu1 = read(model_in[("mu", 1)], continuous)
        mu2 = read(model_in[("mu", 2)], continuous)
        mu = (mu1 + mu2) / 2
        u = mu2 - mu
        write(model_out[("mu", 1)], mu, continuous)
        write(aux_out["u"], u, continuous)

latent_trace = {
    "two_clusters" : 0,
    ("mu", 1) : 1.123
}
observations = {
    ("x", 1) : torch.tensor(1.523),
    ("x", 2) : torch.tensor(1.923),
    ("x", 3) : torch.tensor(1.223),
    ("x", 4) : torch.tensor(1.423)
}

p_args = (4,)

for it in range(1, 100):
    (latent_trace, acc) = involution_mcmc_step(p, p_args, q, f, latent_trace, observations, check=True)
    print(latent_trace, acc)
