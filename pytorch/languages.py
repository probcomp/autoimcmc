import torch
from collections import namedtuple
from itertools import chain

# Helper: inject variables into a function's scope:
from functools import wraps
def inject_variables(context, func):
    @wraps(func)
    def new_func(*args, **kwargs):
        func_globals = func.__globals__
        saved_values = func_globals.copy()
        func_globals.update(context)
        try:
            result = func(*args, **kwargs)
        finally:
            for (var, val) in context.items():
                if var in saved_values:
                    func_globals.update({var: saved_values[var]})
                else:
                    del func_globals[var]
        return result
    return new_func

##############################################
# minimal probabilistic programming language #
##############################################

def trace_and_score(p, *args):
    trace = {}
    lpdf = torch.tensor(0.0)
    def sample(dist, addr):
        nonlocal lpdf
        val = dist.sample()
        trace[addr] = val
        lpdf += dist.log_prob(val)
        return val
    p = inject_variables({"sample" : sample}, p)
    p(*args)
    return (trace, lpdf)

def score(trace, p, *args):
    lpdf = torch.tensor(0.0)
    def sample(dist, addr):
        nonlocal lpdf
        val = trace[addr]
        lpdf += dist.log_prob(val)
        return val
    p = inject_variables({"sample" : sample}, p)
    p(*args)
    return lpdf


###############################################
# minimal differentiable programming language #
###############################################

InvolutionRunnerState = namedtuple("InvolutionRunnerState",
    [   "input_model_trace", "input_aux_trace",
        "output_model_latent_trace", "output_aux_trace",
        "input_cont_tensors", "output_cont_tensors",
        "input_copied_addrs"
    ])

# model and address identifiers
MODEL = "model"
AUX = "aux"

# for syntactic sugar
class ModelSugar:
    def __getitem__(self, key):
        return (MODEL, key)

class AuxSugar:
    def __getitem__(self, key):
        return (AUX, key)

model_in = ModelSugar()
model_out = ModelSugar()
aux_in = AuxSugar()
aux_out = AuxSugar()

def get_which(addr):
    exc = Exception("address argument must be (MODEL, *) or (AUX, *)")
    if len(addr) != 2:
        raise exc
    (which, addr) = addr
    if which != MODEL and which != AUX:
        raise exc
    return (which, addr)

# user-provided type information for reads and writes
continuous = "continuous"
discrete = "discrete"

def check_type_label(type_label):
    if type_label != continuous and type_label != discrete:
        raise Exception("type label argument must be continuous or discrete")

def _read(state, addr, type_label):
    (which, addr) = get_which(addr)
    check_type_label(type_label)
    if type_label == continuous:
        if which == MODEL:
            val = torch.tensor(state.input_model_trace[addr], requires_grad=True)
        else:
            val = torch.tensor(state.input_aux_trace[addr], requires_grad=True)
        state.input_cont_tensors[(which, addr)] = val
        return val
    else:
        if which == MODEL:
            return state.input_model_trace[addr]
        else:
            return state.input_aux_trace[addr]

def _write(state, addr, val, type_label):
    (which, addr) = get_which(addr)
    check_type_label(type_label)
    if type_label == continuous:
        if which == MODEL:
            state.output_model_latent_trace[addr] = val 
        else:
            state.output_aux_trace[addr] = val
        state.output_cont_tensors.append(val)
    else:
        if which == MODEL:
            state.output_model_latent_trace[addr] = val
        else:
            state.output_aux_trace[addr] = val

def _copy(state, addr1, addr2):
    state.input_copied_addrs.append(addr1)
    (which1, addr1) = get_which(addr1)
    (which2, addr2) = get_which(addr2)
    if which1 == MODEL:
        val = state.input_model_trace[addr1]
    else:
        val = state.input_aux_trace[addr1]
    if which2 == MODEL:
        state.output_model_latent_trace[addr2] = val 
    else:
        state.output_aux_trace[addr2] = val
    
def transform_involution(f, state):
    context = {
        "read" : lambda addr, type_label : _read(state, addr, type_label),
        "write" : lambda addr, val, type_label : _write(state, addr, val, type_label),
        "copy" : lambda addr1, addr2 : _copy(state, addr1, addr2)
    }
    new_f = inject_variables(context, f)
    return new_f

def involution_with_jacobian_det(f, input_model_trace, input_auxiliary_trace):
    state = InvolutionRunnerState(input_model_trace, input_auxiliary_trace, {}, {}, {}, [], [])
    f_transformed = transform_involution(f, state)
    f_transformed()
    grads = []
    for output_cont_tensor in state.output_cont_tensors:
        output_cont_tensor.backward(retain_graph=True)
        grad = []
        for (addr, input_cont_tensor) in state.input_cont_tensors.items():
            # skip it if it was copied
            if addr in state.input_copied_addrs:
                continue
            if input_cont_tensor.grad is None:
                grad.append(torch.zeros_like(input_cont_tensor))
            else:
                grad.append(input_cont_tensor.grad.clone())
                input_cont_tensor.grad.zero_()
        grads.append(grad)
    (_, logabsdet) = torch.tensor(grads).slogdet()
    return (state.output_model_latent_trace, state.output_aux_trace, logabsdet)


###################
# involutive MCMC #
###################

def do_involution_check(
        f, observations,
        output_model_latent_trace, output_aux_trace,
        input_model_latent_trace, input_aux_trace):
    output_model_trace = {**output_model_latent_trace, **observations}
    rt_state = InvolutionRunnerState(output_model_trace, output_aux_trace, {}, {}, {}, [], [])
    f_transformed = transform_involution(f, rt_state)
    f_transformed()

    for (addr, val) in rt_state.output_model_latent_trace.items():
        if isinstance(val, torch.Tensor):
            if not torch.eq(val, input_model_latent_trace[addr]):
                raise Exception("involution check failed at model:", addr, val, input_model_latent_trace[addr])
        else:
            if val != input_model_latent_trace[addr]:
                raise Exception("involution check failed model: ", addr, val, input_model_latent_trace[addr])

    for (addr, val) in rt_state.output_aux_trace.items():
        if isinstance(val, torch.Tensor):
            if not torch.allclose(val, input_aux_trace[addr]):
                raise Exception("involution check failed at aux:", addr, val, input_aux_trace[addr])
        else:
            if val != input_aux_trace[addr]:
                raise Exception("involution check failed aux: ", addr, val, input_aux_trace[addr])

    for (addr, val) in input_model_latent_trace.items():
        if isinstance(val, torch.Tensor):
            if not torch.allclose(val, rt_state.output_model_latent_trace[addr]):
                raise Exception("involution check failed at model:", addr, val, rt_state.output_model_latent_trace[addr])
        else:
            if val != rt_state.output_model_latent_trace[addr]:
                raise Exception("involution check failed model: ", addr, val, input_model_latent_trace[addr])

    for (addr, val) in input_aux_trace.items():
        if isinstance(val, torch.Tensor):
            if not torch.allclose(val, rt_state.output_aux_trace[addr]):
                raise Exception("involution check failed at aux:", addr, val, rt_state.output_aux_trace[addr])
        else:
            if val != rt_state.output_aux_trace[addr]:
                raise Exception("involution check failed aux: ", addr, val, rt_state.output_aux_trace[addr])


def involution_mcmc_step(p, p_args, q, f, input_model_latent_trace, observations, check=False):

    # merge latents and observations to form model trace
    input_model_trace = {**input_model_latent_trace, **observations}

    # sample from auxiliary program
    (input_auxiliary_trace, fwd_score) = trace_and_score(q, input_model_trace)

    # run involution
    (output_model_latent_trace, output_auxiliary_trace, logabsdet) = involution_with_jacobian_det(
        f, input_model_trace, input_auxiliary_trace)
    output_model_trace = {**output_model_latent_trace, **observations}

    # do the round trip check
    if check:
        do_involution_check(f, observations,
            output_model_latent_trace, output_auxiliary_trace,
            input_model_latent_trace, input_auxiliary_trace)

    # compute acceptance probability
    prev_score = score(input_model_trace, p, *p_args)
    new_score = score(output_model_trace, p, *p_args)
    bwd_score = score(output_auxiliary_trace, q, output_model_trace)
    prob_accept = min(1, torch.exp(new_score - prev_score + logabsdet + bwd_score - fwd_score))

    # accept or reject
    if torch.distributions.bernoulli.Bernoulli(prob_accept).sample():
        return (output_model_latent_trace, True)
    else:
        return (input_model_latent_trace, False)
