from languages import *

#########################################
# polar vs cartesian coordinate example #
#########################################

Bernoulli = torch.distributions.bernoulli.Bernoulli
Gamma = torch.distributions.gamma.Gamma
Normal = torch.distributions.normal.Normal
Uniform = torch.distributions.uniform.Uniform

pi = 3.1415927410125732

def polar_to_cartesian(r, theta):
    x = torch.cos(r) * theta
    y = torch.sin(r) * theta
    return (x, y)

def cartesian_to_polar(x, y):
    theta = torch.atan2(y, x)
    y = torch.sqrt(x * x + y * y)
    return (theta, y)

def p():
    u = sample(Normal(0, 1), "u")
    v = sample(Normal(0, 1), "v")
    if sample(Bernoulli(0.5), "polar"):
        r = sample(Gamma(1.0, 1.0), "r")
        theta = sample(Uniform(-pi/2, pi/2), "theta")
    else:
        x = sample(Normal(0.0, 1.0), "x")
        y = sample(Normal(0.0, 1.0), "y")
    sample(Normal(0, 1), "z")
    return None

def q(model_trace):
    return None

def f():
    polar = read(model_in["polar"], discrete)
    if polar:
        r = read(model_in["r"], continuous)
        theta = read(model_in["theta"], continuous)
        (x, y) = polar_to_cartesian(r, theta)
        write(model_out["x"], x, continuous)
        write(model_out["y"], y, continuous)
    else:
        x = read(model_in["x"], continuous)
        y = read(model_in["y"], continuous)
        (r, theta) = cartesian_to_polar(x, y)
        write(model_out["r"], r, continuous)
        write(model_out["theta"], theta, continuous)
    write(model_out["polar"], 1 - polar, discrete)
    copy(model_out["u"], (MODEL, "u"))
    u = read(model_in["u"], continuous)
    v = read(model_in["v"], continuous)
    write(model_out["v"], u - v, continuous)

latent_trace = {
        "polar" : 1,
        "r": 1.2,
        "theta" : 0.12,
        "u" : -0.123,
        "v" : 3.31
}

observations = {
        "z" : -0.14
}

for it in range(1, 100):
    (latent_trace, acc) = involution_mcmc_step(p, (), q, f, latent_trace, observations, check=True)
    print(latent_trace, acc)
