import torch
import torch.nn.functional as F

def advanced_disentangle_loss(z_c_detach, z_nc_list, z_c_views, temp=0.1):
    view_num = len(z_nc_list)
    batch_size = z_c_detach.shape[0]

    z_c_norm = F.normalize(z_c_detach, p=2, dim=-1)

    sim_c = torch.matmul(z_c_norm, z_c_norm.T) / temp
    mask = torch.eye(batch_size, device=z_c_detach.device).bool()
    sim_c.masked_fill_(mask, -1e9)
    p_graph_c = F.softmax(sim_c, dim=-1)

    loss_align = 0.0
    loss_ortho = 0.0
    for v in range(view_num):
        z_cv_norm = F.normalize(z_c_views[v], p=2, dim=-1)
        z_nc_v = F.normalize(z_nc_list[v], p=2, dim=-1)

        sim_v = torch.matmul(z_cv_norm, z_cv_norm.T) / temp
        sim_v.masked_fill_(mask, -1e9)
        log_p_graph_v = F.log_softmax(sim_v, dim=-1)
        loss_align += -torch.mean(torch.sum(p_graph_c * log_p_graph_v, dim=-1))

        loss_ortho += cov_ortho_loss(z_c_detach, z_nc_list[v])

    loss_align /= view_num
    loss_ortho /= view_num

    total_loss = (1 * loss_align + 6000 * loss_ortho)
    return total_loss


def cov_ortho_loss(z_c_detach, z_nc):
    x = z_c_detach
    y = z_nc

    x = (x - x.mean(dim=0, keepdim=True))/(x.std(dim=0, keepdim=True)+1e-8)
    y = (y - y.mean(dim=0, keepdim=True))/(y.std(dim=0, keepdim=True)+1e-8)

    n = x.size(0)
    cov_c_nc = torch.matmul(x.T, y) / max(1, n-1)

    d1, d2 = x.size(1), y.size(1)
    loss = torch.norm(cov_c_nc, p="fro")**2 / (d1*d2)
    return loss


def Topological_contrast_loss(args, hidden, nbr_idx, neg_idx, idx, tau=0.2, base_margin=0.01):
    if not getattr(args, 'do_contrast', True):
        return torch.tensor(0., device=hidden.device, dtype=torch.float32)

    device = hidden.device
    if not isinstance(idx, torch.Tensor):
        idx_tensor = torch.tensor(idx, device=device, dtype=torch.long)
    else:
        idx_tensor = idx.long().to(device)

    B = len(idx_tensor)
    if B == 0 or nbr_idx is None or neg_idx is None:
        return torch.tensor(0., device=device, dtype=torch.float32)

    anchor_feat = hidden[idx_tensor]
    pos_indices = nbr_idx[idx_tensor].to(device)
    neg_indices = neg_idx[idx_tensor].to(device)

    pos_feat = hidden[pos_indices].detach()
    neg_feat = hidden[neg_indices].detach()

    anchor_norm = F.normalize(anchor_feat, dim=-1, p=2)
    pos_norm = F.normalize(pos_feat, dim=-1, p=2)
    neg_norm = F.normalize(neg_feat, dim=-1, p=2)

    pos_cos = torch.sum(anchor_norm.unsqueeze(1) * pos_norm, dim=-1)
    neg_cos = torch.sum(anchor_norm.unsqueeze(1) * neg_norm, dim=-1)

    with torch.no_grad():
        anchor_teacher = anchor_norm.detach()
        pos_teacher = pos_norm.detach()
        teacher_sim = torch.sum(anchor_teacher.unsqueeze(1) * pos_teacher, dim=-1)
        sim_mean = teacher_sim.mean(dim=-1, keepdim=True)
        topo_gate = torch.sigmoid((teacher_sim - sim_mean) * 5.0) + 0.0

    anisotropic_margin = base_margin * torch.relu(pos_cos.detach())
    pos_cos_scaled = (pos_cos - anisotropic_margin) / tau
    neg_cos_scaled = neg_cos / tau

    positive = torch.exp(pos_cos_scaled)
    negative = torch.exp(neg_cos_scaled).sum(dim=-1, keepdim=True)

    log_term = -torch.log(positive / (negative + 1e-8) + 1e-8)
    weighted_log_term = (log_term * topo_gate).sum(dim=-1)
    loss_topo = weighted_log_term.sum() / B

    return loss_topo

