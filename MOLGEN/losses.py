import torch
from utils.graph_utils import node_flags, mask_x, mask_adjs, gen_noise
import torch.nn.functional as F

def get_pred_loss_fn(mix_x, mix_adj, train=True, reduce_mean=False, eps=1e-3, lambda_train=1, loss_type=None):
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: torch.sum(*args, **kwargs)

  def compute_loss(pred, target, loss_coeff):
    losses = torch.square( (pred - target) * loss_coeff[:, None, None] ) 
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * 0.5
    return losses

  def get_loss_coeff(sde, t, loss_type):
    if loss_type=='default':
        loss_coeff = sde.loss_coeff(t)
    elif loss_type=='const':
        loss_coeff = torch.ones_like(t)
    else:
        raise NotImplementedError(f'Loss type: {loss_type} not implemented.')
    return loss_coeff

  def loss_fn(model, x, adj, embedding, embedding_x, prior_samples=None, repa_on=False):
    sde_x = mix_x.bridge(x)
    sde_adj = mix_adj.bridge(adj)
    t = torch.rand(adj.shape[0], device=adj.device) * (sde_adj.T - eps) 
    flags = node_flags(adj)

    if prior_samples is None:
        x0 = sde_x.prior_sampling(x.shape, x.device)
        adj0 = sde_adj.prior_sampling_sym(adj.shape, adj.device)
    else:
        x0, adj0 = prior_samples
    x0 = mask_x(x0, flags)
    adj0 = mask_adjs(adj0, flags)

    mean_x, std_x = sde_x.marginal_prob(x0, t)
    xt = mean_x + std_x[:, None, None] * gen_noise(x, flags, sym=False)
    xt = mask_x(xt, flags)

    mean_adj, std_adj = sde_adj.marginal_prob(adj0, t)
    adjt = mean_adj + std_adj[:, None, None] * gen_noise(adj, flags, sym=True) 
    adjt = mask_adjs(adjt, flags)

    valid_counts = torch.max(torch.sum(flags,dim=-1)).int()
    xt, adjt = xt[:,:valid_counts], adjt[:, :valid_counts, :valid_counts]
    x, adj = x[:,:valid_counts], adj[:, :valid_counts, :valid_counts]
    embedding_x, embedding = embedding_x[:, :valid_counts], embedding[:, :valid_counts, :valid_counts]
    flags = flags[:,:valid_counts]
    pred_x, pred_adj, hidden_x, hidden_adj = model(xt, adjt, t.unsqueeze(-1), flags)
    hidden_x_list = list(hidden_x) if isinstance(hidden_x, (list, tuple)) else [hidden_x]
    hidden_adj_list = list(hidden_adj) if isinstance(hidden_adj, (list, tuple)) else [hidden_adj]
    main_idx = 1 if len(hidden_x_list) > 1 else 0

    loss_coeff_x = get_loss_coeff(sde_x, t, loss_type.x)
    loss_coeff_adj = get_loss_coeff(sde_adj, t, loss_type.adj)

    losses_x = compute_loss(pred_x, x, loss_coeff_x)
    losses_adj = compute_loss(pred_adj, adj, loss_coeff_adj)
    losses = torch.mean(losses_x) + torch.mean(losses_adj) * lambda_train
    
    
    repr_by_nodes = None
    if repa_on:
        zero_diag_mask = embedding_x.abs().sum(dim=(1, 2)) == 0
        zero_adj_mask = embedding.abs().sum(dim=(1, 2, 3)) == 0
        cos_diag_mean = torch.zeros(x.shape[0], device=x.device, dtype=hidden_x_list[0].dtype)
        if (~zero_diag_mask).any():
            active_diag = ~zero_diag_mask
            hidden_x_active = hidden_x[active_diag].reshape(hidden_x[active_diag].shape[0], -1)
            embedding_x_active = embedding_x[active_diag].reshape(embedding_x[active_diag].shape[0], -1)
            node_counts = flags[active_diag].sum(dim=1).to(loss_coeff_x)
            cos_diag_mean[active_diag] = -F.cosine_similarity(hidden_x_active, embedding_x_active, dim=-1) * node_counts#loss_coeff_x

        cos_adj_mean = torch.zeros(x.shape[0], device=x.device, dtype=hidden_adj_list[0].dtype)
        if (~zero_adj_mask).any():
            active_adj = ~zero_adj_mask
            hidden_adj_active = hidden_adj[active_adj].reshape(hidden_adj[active_adj].shape[0], -1)
            embedding_adj_active = embedding[active_adj].reshape(embedding[active_adj].shape[0], -1)
            node_counts = flags[active_adj].sum(dim=1).to(loss_coeff_x)
            node_counts_sq = node_counts * node_counts
            cos_adj_mean[active_adj] = -F.cosine_similarity(hidden_adj_active, embedding_adj_active, dim=-1) * node_counts_sq#loss_coeff_adj

    return (losses, torch.mean(losses_x).detach().detach(), torch.mean(losses_adj).detach().detach(),
            torch.mean(cos_diag_mean), torch.mean(cos_adj_mean) * lambda_train)
  return loss_fn






def cknna(
    v,
    targets,
    k_per
):
    k_per_tuple = (float(k_per),) if isinstance(k_per, (float, int)) else tuple(k_per)
    v = F.normalize(v, p=2, dim=-1)
    e = F.normalize(targets, p=2, dim=-1)

    bs = v.shape[0]
    diag = torch.arange(bs, device=v.device)

    dist_v = -v @ v.T
    dist_e = -e @ e.T
    dist_v[diag, diag] = float("inf")
    dist_e[diag, diag] = float("inf")
    sim_v = -dist_v
    sim_e = -dist_e

    sim_v[diag, diag] = -torch.inf
    sim_e[diag, diag] = -torch.inf

    k_values = [max(1, int(float(bs) * kp)) for kp in k_per_tuple]
    k_max = max(k_values)
    _, topk_v = sim_v.topk(k_max, dim=-1)
    _, topk_e = sim_e.topk(k_max, dim=-1)

    neigh_v = torch.zeros(bs, bs, dtype=torch.bool, device=v.device)
    neigh_e = torch.zeros_like(neigh_v)

    scores = []
    row_idx_full = torch.arange(bs, device=v.device).unsqueeze(1)
    for k in k_values:
        row_idx = row_idx_full.expand(-1, k)
        neigh_v.zero_()
        neigh_e.zero_()
        neigh_v[row_idx, topk_v[:, :k]] = True
        neigh_e[row_idx, topk_e[:, :k]] = True

        inter_counts = (neigh_v & neigh_e).sum(dim=-1)
        overlaps = inter_counts.float() / float(k)
        scores.append(overlaps.mean().item())
    return scores[0] if len(scores) == 1 else tuple(scores)
