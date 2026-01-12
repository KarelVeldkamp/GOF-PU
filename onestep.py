from numpy.polynomial.hermite import hermgauss
import torch
import torch.nn as nn
import time

# function used to create quadrature nodes
def create_integration_nodes(nquad, ndim, n):
    nodes, weights = hermgauss(nquad)

    # Move to the same device as the model's parameters
    nodes = torch.tensor(nodes, dtype=torch.float32) * torch.sqrt(torch.tensor(2.0))  # Adjust for standard normal
    weights = (torch.tensor(weights, dtype=torch.float32) / torch.sqrt(torch.tensor(torch.pi)))  # Normalize weights
    # Create all possible indices for combinations of length 3

    if ndim == 1:
        cnodes = nodes.unsqueeze(1)
        cweights = weights
    else:
        tensors = [torch.arange(nquad) for i in range(ndim)]
        # Use the unpacking operator * to pass them to torch.cartesian_prod

        indices = torch.cartesian_prod(*tensors)
        # Use advanced indexing to get the combinations
        cnodes = nodes[indices]
        cweights = weights[indices].prod(-1)
    cnodes = cnodes.unsqueeze(1).repeat(1, n, 1)

    return cnodes, cweights

# function ised to compute marginal log likelihood
def mlll(params, cnodes, cweights, Q, data):
    # params: [n_slopes + n_intercepts]
    # cnodes: [n_quad_points, n_people, n_factors] (3375, 1000, 3)
    # cweights: [n_quad_points] (3375,)
    # Q: [n_factors, n_items] (3, items)
    # data: [n_people, n_items] (1000, items)

    n_weights = Q.sum().int()
    n_items = Q.shape[1]

    # 1. Reconstruct parameters with probit scaling (×1.702 for logit→probit)
    slopes = torch.zeros(*Q.T.shape).to(params)  # [items, factors]
    slopes[Q.T.bool()] = params[:n_weights]
    bias = params[n_weights:n_weights + n_items]  # [items,]

    # 2. Compute probabilities for all combinations
    # einsum explanation:
    # 'qpf,if->qpi' = quad_point × person × factor, item × factor → quad × person × item
    logits = torch.einsum('qpf,if->qpi', cnodes, slopes) + bias
    probs = torch.sigmoid(logits)

    # 3. Log-likelihood with safe log
    log_probs = torch.where(data > 0,
                            torch.log(probs.clamp(min=1e-10)),
                            torch.log((1 - probs).clamp(min=1e-10)))  # [3375, 1000, items]

    # 4. Sum over items and add log weights
    item_ll = log_probs.sum(-1)  # [3375, 1000]
    weighted_ll = item_ll + torch.log(cweights).unsqueeze(1)  # [3375, 1000]

    # 5. Marginalize over quadrature points
    return torch.logsumexp(weighted_ll, dim=0)  # sum over people

def onestep(vae, data, nquad, se_type='fisher', batch_size=None):
    """
    function used to perform one step of fisher scoring
    :param vae: pytorch module containing model estimated using AVI
    :param data: torch tensor containing the binary data
    :param nquad: number of quadrature points to use for numerical integration
    :param batch_size: batch size to use for computing observed information matrix (default is 1)
    :return: (new a parameters, new b parameters, standard error for a, standard error for b,
    sandwich estiamte for a, sandwich estimate for b).
        Please note that the standard errors are computed before doing the one step, so in order to compute standard
        errors after the onestep procedure this function needs to be called twice.
    """


    # --- added: pick device and move inputs / params there ---
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data = data.to(dev)
    W = vae.decoder.weights.to(dev)
    b = vae.decoder.bias.to(dev)

    if batch_size == None:
        batch_size = data.shape[0]

    # create vector of model parameters
    est_params = nn.Parameter(torch.cat((W.T[W.T != 0], b.T)))
    # Assume weights_shape and bias_size are known
    n_weights = vae.decoder.qm.sum().int()
    n_items = vae.decoder.qm.shape[1]

    cnodes, cweights = create_integration_nodes(nquad, vae.latent_dims, data.shape[0])
    cnodes, cweights = cnodes.to(dev), cweights.to(dev)

    # make Q from device copy of weights
    Q = (W != 0).float()

    # compute the gradient
    t1 = time.time()
    grad = torch.autograd.grad(
        mlll(est_params, cnodes, cweights, Q, data).sum(),  # call the function
        est_params,
        create_graph=True
    )[0]

    bnodes, bweights = create_integration_nodes(nquad, vae.latent_dims, batch_size)
    bnodes, bweights = bnodes.to(dev), bweights.to(dev)

    outer_pp = torch.zeros((data.shape[0], est_params.shape[0], est_params.shape[0]), device=dev)

    for i in range(data.shape[0] // batch_size):
        batch = data[(i * batch_size):((i + 1) * batch_size), :]

        batch_grad = torch.autograd.functional.jacobian(
            lambda p: mlll(p, bnodes, bweights, Q, batch),
            est_params,
            create_graph=False
        ).unsqueeze(-1)

        outer = batch_grad @ batch_grad.transpose(2, 1)

        outer_pp[(i * batch_size):((i + 1) * batch_size), :, :] = outer.detach()

    FI = outer_pp.sum(0)
    I_FI = torch.eye(FI.shape[0], device=FI.device, dtype=FI.dtype)
    FI_inv = torch.linalg.inv(FI + I_FI * 1e-6)

    if se_type == 'sandwich':
        H = torch.autograd.functional.hessian(
            lambda p: mlll(p, cnodes, cweights, Q, data).sum(),
            est_params
        )
        # Conventionally, use -H if mlll is a log-likelihood (since H is negative definite at the max).
        H = -H
        I_H = torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
        H_inv = torch.linalg.inv(H + I_H * 1e-6)
        V_robust = H_inv @ FI @ H_inv  # FI is your B
        se = torch.sqrt(torch.diag(V_robust))
    elif se_type == 'fisher':
        se = torch.sqrt(torch.diag(FI_inv))
    else:
        raise ValueError('se_type must be either "sandwich" or "fisher"')
    # update parameters
    newpars = est_params.clone() + FI_inv @ grad
    # compute standard errors


    new_a = torch.zeros_like(W.T)
    se_a = torch.zeros_like(W.T)
    se_robust_a = torch.zeros_like(W.T)

    new_a[Q.T == 1] = newpars[:Q.int().sum()]
    se_a[Q.T == 1] = se[:Q.int().sum()]
    new_b = newpars[Q.int().sum():]
    se_b = se[Q.int().sum():]


    return new_a.cpu().detach(), new_b.cpu().detach(), se_a.cpu().detach(), se_b.cpu().detach()