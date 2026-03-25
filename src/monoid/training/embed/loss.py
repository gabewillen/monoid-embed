import torch
import torch.nn as nn
import torch.nn.functional as F

class GeometricDistillationLoss(nn.Module):
    """
    Minimizes the distance between student and teacher embeddings.
    Combines Cosine Similarity Loss and MSE Loss.
    """
    def __init__(self, alpha_mse=0.0):
        super().__init__()
        self.alpha_mse = alpha_mse
        self.cosine_loss = nn.CosineEmbeddingLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, student_emb, teacher_emb, detach_teacher: bool = True):
        # Teacher embeddings are targets, detach them just in case
        if detach_teacher:
            teacher_emb = teacher_emb.detach()
        
        # Cosine Loss: Expects inputs to be (x1, x2, target). Target 1 means similar.
        # We want maximize similarity => minimize 1 - cos(x, y).
        # nn.CosineEmbeddingLoss(x, y, 1) computes 1 - cos(x,y).
        target = torch.ones(student_emb.shape[0], device=student_emb.device)
        loss_cos = self.cosine_loss(student_emb, teacher_emb, target)
        
        loss = loss_cos
        
        if self.alpha_mse > 0:
            loss_mse = self.mse_loss(student_emb, teacher_emb)
            loss += self.alpha_mse * loss_mse
            
        return loss

class SpreadOutRegularizer(nn.Module):
    """
    Encourages embeddings to be uniformly distributed on the unit sphere.
    Based on 'Spread-Out Regularization' (Zhang et al.)
    """
    def __init__(self):
        super().__init__()

    def forward(self, embeddings):
        # embeddings: (B, D), assumed normalized
        # Compute pairwise dot products
        # (B, D) @ (D, B) -> (B, B)
        # We want to minimize the squared dot products of distinct pairs
        
        batch_size = embeddings.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=embeddings.device)
            
        gram_matrix = torch.mm(embeddings, embeddings.t())
        
        # Remove diagonals (self-similarity is always 1)
        # We can just subtract the identity matrix since inputs are normalized
        off_diag = gram_matrix - torch.eye(batch_size, device=embeddings.device)
        
        # Minimize sum of squares of off-diagonal elements
        loss = (off_diag ** 2).sum() / (batch_size * (batch_size - 1))
        
        return loss

class HardnessWeightedContrastiveLoss(nn.Module):
    """
    InfoNCE-style loss with hardness weighting.
    w_i = exp(alpha * stop_grad(sim(q, n)))
    """
    def __init__(self, temperature=0.07, alpha_hardness=5.0):
        super().__init__()
        self.temperature = temperature
        self.alpha_hardness = alpha_hardness

    def forward_distillation(self, student_emb, teacher_emb, assume_normalized: bool = False):
        # student: (B, D), teacher: (B, D)
        # Calculates similarity matrix (B, B)
        # logits[i, j] = sim(student[i], teacher[j]) / temp
        
        # Normalize
        if not assume_normalized:
            student_emb = F.normalize(student_emb, p=2, dim=-1)
            teacher_emb = F.normalize(teacher_emb, p=2, dim=-1)
        teacher_emb = teacher_emb.detach()
        
        sim_matrix = torch.mm(student_emb, teacher_emb.t()) # (B, B)
        logits = sim_matrix / self.temperature
        
        batch_size = student_emb.shape[0]
        
        # Hardness Weighting
        # w_ij = exp(alpha * sg(sim(s_i, t_j)))
        # We want to UPWEIGHT high-similarity negatives (hard negatives).
        
        with torch.no_grad():
            hardness_weights = torch.exp(self.alpha_hardness * sim_matrix)
            mask = torch.eye(batch_size, device=student_emb.device)
            hardness_weights = hardness_weights * (1 - mask) + mask # keep Pos weight 1.0
        
        # Denominator: sum(w * exp(logits))
        exp_logits = torch.exp(logits)
        weighted_exp_logits = hardness_weights * exp_logits
        denominator = weighted_exp_logits.sum(dim=1) # (B,)
        
        # Numerator: exp(sim(s_i, t_i)/T) -> which is exp(logits.diag())
        numerator = exp_logits.diag()
        
        # Prob = Num / Denom
        # Loss = -log(Prob) = -log(Num) + log(Denom)
        loss = -torch.log(numerator) + torch.log(denominator)
        
        return loss.mean()

class ConsistencyLoss(nn.Module):
    """
    Minimizes distance between Causal and Bidirectional embeddings.
    'State Re-anchoring' training objective.
    """
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, causal_emb, bidirectional_emb):
        # bidirectional_emb acts as the target (detached)
        target = bidirectional_emb.detach()
        return self.mse_loss(causal_emb, target)

class PairwiseCosineDistillationLoss(nn.Module):
    """
    Matches student pairwise cosine similarities to teacher pairwise cosine similarities.
    Operates on off-diagonal entries only.
    """
    def __init__(self):
        super().__init__()

    def forward(self, student_emb, teacher_emb, assume_normalized: bool = False, detach_teacher: bool = True):
        if student_emb.size(0) <= 1:
            return torch.tensor(0.0, device=student_emb.device)
        if assume_normalized:
            student_normed = student_emb
            teacher_normed = teacher_emb.detach() if detach_teacher else teacher_emb
        else:
            student_normed = F.normalize(student_emb, p=2, dim=-1)
            teacher_normed = F.normalize(teacher_emb.detach() if detach_teacher else teacher_emb, p=2, dim=-1)
        student_cos = student_normed @ student_normed.t()
        teacher_cos = teacher_normed @ teacher_normed.t()
        mask = ~torch.eye(student_cos.size(0), dtype=torch.bool, device=student_cos.device)
        return F.mse_loss(student_cos[mask], teacher_cos[mask])

class RKDDistanceLoss(nn.Module):
    """
    Relational Knowledge Distillation distance loss.
    Matches normalized pairwise L2 distances between student and teacher.
    """
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, student_emb, teacher_emb, detach_teacher: bool = True):
        if student_emb.size(0) <= 1:
            return torch.tensor(0.0, device=student_emb.device)
        if detach_teacher:
            teacher_emb = teacher_emb.detach()
        s_dist = torch.cdist(student_emb, student_emb, p=2)
        t_dist = torch.cdist(teacher_emb, teacher_emb, p=2)
        mask = ~torch.eye(s_dist.size(0), dtype=torch.bool, device=s_dist.device)
        s_dist = s_dist[mask]
        t_dist = t_dist[mask]
        s_dist = s_dist / s_dist.mean().clamp_min(self.eps)
        t_dist = t_dist / t_dist.mean().clamp_min(self.eps)
        return F.smooth_l1_loss(s_dist, t_dist)

class RKDAngleLoss(nn.Module):
    """
    Relational Knowledge Distillation angle loss.
    Matches angles between difference vectors within a batch.
    """
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, student_emb, teacher_emb, detach_teacher: bool = True):
        if student_emb.size(0) <= 2:
            return torch.tensor(0.0, device=student_emb.device)
        if detach_teacher:
            teacher_emb = teacher_emb.detach()
        t_diffs = teacher_emb.unsqueeze(0) - teacher_emb.unsqueeze(1)
        s_diffs = student_emb.unsqueeze(0) - student_emb.unsqueeze(1)
        t_diffs = F.normalize(t_diffs, p=2, dim=-1, eps=self.eps)
        s_diffs = F.normalize(s_diffs, p=2, dim=-1, eps=self.eps)
        t_angles = torch.bmm(t_diffs, t_diffs.transpose(1, 2))
        s_angles = torch.bmm(s_diffs, s_diffs.transpose(1, 2))
        return F.smooth_l1_loss(s_angles, t_angles)

class SimilarityPreservingKDLoss(nn.Module):
    """
    Matches student and teacher similarity matrices within a batch.
    """
    def __init__(self):
        super().__init__()

    def forward(self, student_emb, teacher_emb, assume_normalized: bool = False, detach_teacher: bool = True):
        if student_emb.size(0) <= 1:
            return torch.tensor(0.0, device=student_emb.device)
        if assume_normalized:
            student_normed = student_emb
            teacher_normed = teacher_emb.detach() if detach_teacher else teacher_emb
        else:
            student_normed = F.normalize(student_emb, p=2, dim=-1)
            teacher_normed = F.normalize(teacher_emb.detach() if detach_teacher else teacher_emb, p=2, dim=-1)
        gram_student = student_normed @ student_normed.t()
        gram_teacher = teacher_normed @ teacher_normed.t()
        return F.mse_loss(gram_student, gram_teacher)

class VICRegVarianceLoss(nn.Module):
    """
    Penalizes per-dimension variance below a floor to prevent collapse.
    """
    def __init__(self, variance_floor: float = 1.0):
        super().__init__()
        self.variance_floor = variance_floor

    def forward(self, student_emb):
        if student_emb.size(0) <= 1:
            return torch.tensor(0.0, device=student_emb.device)
        std = student_emb.std(dim=0)
        return F.relu(self.variance_floor - std).mean()

class NeighborhoodDistillationLoss(nn.Module):
    """
    Matches student query->doc similarity distributions to teacher distributions.
    """
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_queries, student_docs, teacher_queries, teacher_docs):
        if student_queries.numel() == 0 or student_docs.numel() == 0:
            return torch.tensor(0.0, device=student_queries.device)

        student_queries = F.normalize(student_queries, p=2, dim=-1)
        student_docs = F.normalize(student_docs, p=2, dim=-1)
        teacher_queries = F.normalize(teacher_queries.detach(), p=2, dim=-1)
        teacher_docs = F.normalize(teacher_docs.detach(), p=2, dim=-1)

        t_logits = (teacher_queries @ teacher_docs.t()) / self.temperature
        s_logits = (student_queries @ student_docs.t()) / self.temperature

        t_probs = F.softmax(t_logits, dim=-1)
        s_log_probs = F.log_softmax(s_logits, dim=-1)
        return F.kl_div(s_log_probs, t_probs, reduction="batchmean")
