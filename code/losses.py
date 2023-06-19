import torch
import torch.nn as nn
import einops
import torch.nn.functional as F

"""
All losses take in a dict containing network outputs and other 
values needed (logits, features, statistics etc.) 
and the target labels.

For each baseline method, we have a specific loss operating on scenarios and one operating on locations.
We implement this by creating a base class for each.
Subclasses specific for scenario/location can inherit from this base class and override the `__call__` method to implement specific loss calculations.

"""


class CIR_text_loss(nn.Module):
    """
    Standard video-text contrastive loss (CLIP-based).
    """

    def __init__(self, config, dataset=None):
        super().__init__()
        self.loss_img = nn.CrossEntropyLoss()
        self.loss_text = nn.CrossEntropyLoss()
        self.device = torch.device('cuda')
        self.config = config

    def __call__(self, output, target):
        logits_per_image_video = output["logits_video"]
        logits_per_text_video = output["logits_txt"]

        ground_truth = torch.arange(len(target), dtype=torch.long, device=self.device)
        total_loss_verb = (self.loss_img(logits_per_image_video, ground_truth) + self.loss_text(logits_per_text_video,
                                                                                                ground_truth)) / 2

        return total_loss_verb


class CE_base(nn.Module):
    """
    Standard cross entropy loss.
    """

    def __init__(self, index):
        super().__init__()
        self.index = index
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, output, target):
        target = target[:, self.index]
        return self.loss(output, target.long())


class CE(CE_base):
    """
    Standard cross entropy loss for actions.
    """

    def __init__(self, config, dataset=None):
        super(CE, self).__init__(index=0)


class CE_S(CE_base):
    """
    Standard cross entropy loss for scenarios.
    """

    def __init__(self, config, dataset=None):
        super(CE_S, self).__init__(index=1)


class CE_L(CE_base):
    """
    Standard cross entropy loss for locations.
    """

    def __init__(self, config, dataset=None):
        super(CE_L, self).__init__(index=2)


class CE_DoPrompt_adapt(nn.Module):
    """
    Adaptation loss for training domain prompts.
    """

    def __init__(self, config, index):
        super().__init__()
        self.index = index
        self.config = config
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, output, target):
        target = target[:, self.index]
        logits_adapt = output
        gt = torch.zeros(logits_adapt.shape[:-1], dtype=torch.float32, device=logits_adapt.device)
        gt.scatter_(1, target.unsqueeze(1).to(torch.int64), 1)
        gt = gt.unsqueeze(-1).repeat(1, 1, self.config.prompt_dim)
        loss_adapt = F.binary_cross_entropy(logits_adapt, gt)
        return loss_adapt


class CE_DoPrompt_adapt_S(CE_DoPrompt_adapt):
    """
        Adaptation loss for training prompts for scenarios.
    """

    def __init__(self, config, dataset=None):
        super(CE_DoPrompt_adapt_S, self).__init__(config, index=1)


class CE_DoPrompt_adapt_L(CE_DoPrompt_adapt):
    """
        Adaptation loss for training prompts for locations.
    """

    def __init__(self, config, dataset=None):
        super(CE_DoPrompt_adapt_L, self).__init__(config, index=2)


class BODA(nn.Module):
    """
    BODA on representations.
    """

    def __init__(self, config, dataset, index):
        super().__init__()
        self.config = config
        self.index = index
        self.n_domains = config.n_scenarios if self.index == 1 else config.n_locations
        self.CE_loss = nn.CrossEntropyLoss()
        self.class_domain_counts = nn.parameter.Parameter(
            data=dataset.get_class_scenario_counts() if self.index == 1 else dataset.get_class_location_counts(),
            requires_grad=False).cuda()
        self.use_c_d_mask = nn.parameter.Parameter(data=torch.where(self.class_domain_counts > 0, 1, 0),
                                                   requires_grad=False).cuda()

        d_lambda = torch.zeros([config.n_classes, self.n_domains, config.n_classes, self.n_domains])

        for c1 in range(config.n_classes):
            for d1 in range(self.n_domains):
                for c2 in range(config.n_classes):
                    for d2 in range(self.n_domains):
                        d_lambda[c1, d1, c2, d2] = self.class_domain_counts[c1, d1] / (
                                self.class_domain_counts[c2, d2] + 1)
        self.d_lambda = nn.parameter.Parameter(data=torch.pow(d_lambda, self.config.boda_upsilon),
                                               requires_grad=False).cuda()

    def __call__(self, epoch, representation, c_d_means, target):
        """
        Input: dict containing representations for BODA, class statistics and epoch number
        target are the labels of shape batch x 2+, where dim 0 is the class, and dim 1 is the scenario, and dim 2 is the location
        """
        if epoch < self.config.stat_update_start_epoch:
            return torch.tensor(0.0, device=representation.device)

        l_class = target[:, 0]
        l_dom = target[:, self.index]
        l_c_d = target[:, :2] if self.index == 1 else target[:, [0, 2]]
        reps = F.normalize(representation, dim=-1)
        means = F.normalize(c_d_means, dim=-1)
        num_d = len(torch.unique(l_dom))
        n_batch = reps.shape[0]
        device = reps.device
        epsilon = 1e-8

        # calculate distance of every z to every c,d representation
        if self.config.boda_dist == "mean":
            dists = einops.rearrange(reps, 'b d -> b 1 1 d') - means
            dists = torch.norm(dists, dim=-1)


        elif self.config.boda_dist == "mahalanobis":
            covs = output["c_d_covs"]
            dists = einops.rearrange(reps, 'b d -> b 1 1 d') - means
            dists = einops.rearrange(dists, 'b c d dim -> b c d dim 1')
            cov_inv = torch.linalg.pinv(covs)
            m = torch.matmul(cov_inv, dists)
            m = torch.matmul(dists.transpose(-2, -1), m)
            dists = torch.sqrt(m).squeeze()
        else:
            raise NotImplementedError()

        if self.config.boda_balance_domain:
            dists = dists / (self.class_domain_counts.cuda() + 1)
        # one hot masks for each samples class and domain
        z_c_mask = torch.nn.functional.one_hot(l_class.long(), self.config.n_classes).float()
        z_d_mask = torch.nn.functional.one_hot(l_dom.long(), self.n_domains).float()

        if self.config.boda_calibrate_distance:
            # z_lambda are lambda values for every z_i and d, given c_i and d_i: z x d
            zdm = einops.rearrange(z_d_mask, 'b l -> b 1 1 1 l')
            zcm = einops.rearrange(z_c_mask, 'b l -> b 1 1 1 l')
            zcdc = torch.matmul(zdm, self.d_lambda.transpose(-2, -1))
            zcd = torch.matmul(zcm, zcdc.transpose(-2, -1))
            z_lambda = zcd.squeeze()

            # lambda multiplication and exp
            exp_dists = torch.exp(- torch.mul(z_lambda, dists))
        else:
            exp_dists = torch.exp(- dists)

        exp_dists = torch.mul(exp_dists, self.use_c_d_mask)
        exp_dists_sum = torch.sum(exp_dists, dim=[1, 2])

        # numerators: z x d
        numerator = torch.mul(exp_dists, einops.repeat(z_c_mask, 'b c -> b c d', d=self.n_domains))
        numerator = torch.sum(numerator, dim=1)

        # denominators: z x d
        v = torch.ones(n_batch, device=device)
        z_idxs = torch.arange(n_batch, device=device).reshape(-1, 1)
        l_z_c_d = torch.cat([z_idxs, l_c_d], dim=-1)
        z_c_d_mask = torch.sparse_coo_tensor(l_z_c_d.t(), v,
                                             [n_batch, self.config.n_classes, self.n_domains]).to_dense()
        denominator = einops.repeat(exp_dists_sum, 'b -> b d', d=self.n_domains) - torch.sum(
            torch.mul(z_c_d_mask, exp_dists), dim=1) + epsilon

        # numerators / denominators: z x d
        fracs = numerator / denominator + epsilon

        # log n/d z x d
        log_fracs = torch.log(fracs)

        # set elements where d==l_dom[z] to zero: z x d
        z_d_mask_inv = - (z_d_mask - 1)
        log_fracs = torch.mul(log_fracs, z_d_mask_inv)

        # sum
        loss = torch.sum(log_fracs)

        # scale by number of domains
        boda_loss = loss * (-1) / (num_d - 1)

        boda_loss = boda_loss / n_batch

        return boda_loss


class BODA_S(BODA):
    """
    BODA implementation for scenarios
    """

    def __init__(self, config, dataset=None):
        super(BODA_S, self).__init__(config, dataset, index=1)


class BODA_L(BODA):
    """
    BODA implementation for locations
    """

    def __init__(self, config, dataset=None):
        super(BODA_L, self).__init__(config, dataset, index=2)


class AbstractMMD(nn.Module):
    """
       Perform ERM while matching the pair-wise domain feature distributions
       using MMD (abstract class)
       Code adapted from DomainBed (https://github.com/facebookresearch/DomainBed/tree/main)
    """

    def __init__(self, config, kernel, index):
        super().__init__()
        self.config = config
        self.index = index
        self.n_domains = config.n_scenarios if self.index == 1 else config.n_locations
        self.device = torch.device('cuda')
        self.kernel_type = kernel

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        elif self.kernel_type == 'linear':
            delta = x.mean(0, keepdim=True) - y.mean(0, keepdim=True)
            return torch.linalg.norm(delta)
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1 + 1e-12)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1 + 1e-12)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def __call__(self, output, target):

        l_dom = target[:, self.index]
        reps = output.to(self.device)

        loss = 0

        for i in range(self.n_domains):
            for j in range(i + 1, self.n_domains):

                idxs_i = torch.where(l_dom == i, 1, 0).to(self.device)
                idxs_i = torch.nonzero(idxs_i)[:, 0]
                if len(idxs_i) == 0:
                    continue
                idxs_j = torch.where(l_dom == j, 1, 0).to(self.device)
                idxs_j = torch.nonzero(idxs_j)[:, 0]
                if len(idxs_j) == 0:
                    continue

                x_i = torch.index_select(input=reps, dim=0, index=idxs_i).to(self.device)
                x_j = torch.index_select(input=reps, dim=0, index=idxs_j).to(self.device)

                loss += self.mmd(x_i, x_j)

        if (self.n_domains) > 1:
            penalty /= ((self.n_domains) * ((self.n_domains) - 1) / 2)

        return loss


class MMD_gaussian_S(AbstractMMD):
    """
    MMD using Gaussian kernel for scenarios.
    """

    def __init__(self, config, dataset=None):
        super(MMD_gaussian_S, self).__init__(config, kernel='gaussian', index=1)


class MMD_gaussian_L(AbstractMMD):
    """
    MMD using Gaussian kernel for locations.
    """

    def __init__(self, config, dataset=None):
        super(MMD_gaussian_L, self).__init__(config, kernel='gaussian', index=2)


class CORAL_S(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, config, dataset=None):
        super(CORAL_S, self).__init__(config, kernel='cov', index=1)


class CORAL_L(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, config, dataset=None):
        super(CORAL_L, self).__init__(config, kernel='cov', index=2)


class Mixup(nn.Module):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """

    def __init__(self, config, dataset=None):
        super().__init__()
        self.config = config

    def __call__(self, x, y_a, y_b, lam):
        objective = 0
        objective += lam * F.cross_entropy(x, y_a.long())
        objective += (1 - lam) * F.cross_entropy(x, y_b.long())
        objective /= x.size(0)
        return objective
