import torch
import torch.nn as nn
import einops
import math
import torch.nn.functional as F
import numpy as np
"""
All losses take in a dict containing network outputs and other 
values needed (logits, features, statistics etc.)
and the target labels
"""


class GEN(nn.Module):
    """
    Generative caption loss.
    """

    def __init__(self, config, dataset=None):
        super().__init__()
        self.config = config

    def forward(self, output, target=None):
        return output


class CLIP(nn.Module):
    """
    Standard CLIP loss.
    """

    def __init__(self, config, dataset=None):
        super().__init__()
        self.loss_img = nn.CrossEntropyLoss()
        self.loss_text = nn.CrossEntropyLoss()
        self.device = torch.device('cuda')
        self.config = config

    def __call__(self, output, target):
        logits_per_image_video = output["logits_img_video"]
        logits_per_text_video = output["logits_txt_video"]
        logits_per_image_text = output["logits_img_text"]
        logits_per_text_text = output["logits_txt_text"]
        logits_per_image_ori = output["logits_img_ori"]
        logits_per_text_ori = output["logits_txt_ori"]

        ground_truth = torch.arange(len(target), dtype=torch.long, device=self.device)
        total_loss_verb = (self.loss_img(logits_per_image_video, ground_truth) + self.loss_text(logits_per_text_video,
                                                                                                ground_truth)) / 2 if self.config.mix_video else 0
        total_loss_verb_text = (self.loss_img(logits_per_image_text, ground_truth) + self.loss_text(
            logits_per_text_text,
            ground_truth)) / 2 if self.config.mix_txt else 0
        total_loss_verb_ori = (self.loss_img(logits_per_image_ori, ground_truth) + self.loss_text(logits_per_text_ori,
                                                                                                  ground_truth)) / 2 if self.config.ori else 0
        return total_loss_verb + self.config.weight_ori * total_loss_verb_ori + total_loss_verb_text


class CLIP_ori(nn.Module):
    """
    Standard CLIP loss.
    """

    def __init__(self, config, dataset=None):
        super().__init__()
        self.loss_img = nn.CrossEntropyLoss()
        self.loss_text = nn.CrossEntropyLoss()
        self.device = torch.device('cuda')
        self.config = config

    def __call__(self, output, target):
        logits_per_image_video = output["logits_img_video"]
        logits_per_text_video = output["logits_txt_video"]
        logits_per_image_text = output["logits_img_text"]
        logits_per_text_text = output["logits_txt_text"]
        logits_per_image_ori = output["logits_img_ori"]
        logits_per_text_ori = output["logits_txt_ori"]

        ground_truth = torch.arange(len(target), dtype=torch.long, device=self.device)

        total_loss_verb_ori = (self.loss_img(logits_per_image_ori, ground_truth) + self.loss_text(logits_per_text_ori,
                                                                                                  ground_truth)) / 2
        return total_loss_verb_ori


class CustomCLIP(nn.Module):
    """
    Standard CLIP loss.
    """

    def __init__(self, config, dataset=None):
        super().__init__()
        self.config = config
        self.loss_img = nn.CrossEntropyLoss()
        self.device = torch.device('cuda')

    def __call__(self, output, target):
        logits_per_image = output["CustomCLIP"]

        total_loss = self.loss_img(logits_per_image, self.config.n_classes * target[:, 1] + target[:, 0])
        return total_loss


class CE(nn.Module):
    """
    Standard cross entropy loss.
    """

    def __init__(self, config, dataset=None):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, output, target):
        logits = output
        return self.loss(logits, target[:, 0].long())




class CE_source(nn.Module):
    """
    Standard cross entropy loss.
    """

    def __init__(self, config, dataset=None):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, output, target):
        logits = output
        return self.loss(logits, target[:, 2].long())


class CE_scenario(nn.Module):
    """
    Standard cross entropy loss.
    """

    def __init__(self, config, dataset=None):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, output, target):
        logits = output
        return self.loss(logits, target[:, 1].long())


class CE_D_S(nn.Module):
    """
    Standard cross entropy loss.
    """

    def __init__(self, config, dataset=None):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, output, target):
        logits = output
        return self.loss(logits, target[:, 1] * target[:, 2].long())


class CE_D(nn.Module):
    """
    Standard cross entropy loss.
    """

    def __init__(self, config, dataset=None):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, output, target):
        logits = output
        return self.loss(logits, target[:, 1].long())


class CE_S(nn.Module):
    """
    Standard cross entropy loss.
    """

    def __init__(self, config, dataset=None):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, output, target):
        logits = output
        return self.loss(logits, target[:, 2].long())


class CE_DoPrompt_adapt(nn.Module):
    """
    Standard cross entropy loss.
    """

    def __init__(self, config, dataset=None):
        super().__init__()
        self.config = config
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, output, target, is_source=False):
        target = target[:, 1] if not is_source else target[:, 2]

        logits_adapt = output
        gt = torch.zeros(logits_adapt.shape[:-1], dtype=torch.float32, device=logits_adapt.device)
        # gt = F.one_hot(target[:, 1], self.config.n_domains).to(logits_adapt.device)
        gt.scatter_(1, target.unsqueeze(1).to(torch.int64), 1)
        gt = gt.unsqueeze(-1).repeat(1, 1, self.config.prompt_dim)

        loss_adapt = F.binary_cross_entropy(logits_adapt, gt)
        return loss_adapt


class CE_DoPrompt_mix(nn.Module):
    """
    Standard cross entropy loss.
    """

    def __init__(self, config, dataset=None):
        super().__init__()
        self.config = config
        self.loss = nn.CrossEntropyLoss() if self.config.CE_reconstruction else nn.MSELoss()

    def __call__(self, output, mix_weights, target, is_source=False):
        target = target[:, 1] if not is_source else target[:, 2]
        n = self.config.n_domains if not is_source else self.config.n_sources
        scenario_oh = F.one_hot(target.to(torch.int64), num_classes=n)
        domain_sums = torch.matmul(mix_weights, scenario_oh.float())
        domain_wieghts = domain_sums / (torch.sum(domain_sums, dim=0) + 1e-8)
        if self.config.softmax_mix:
            domain_sums = F.softmax(domain_wieghts, dim=1)

        return self.loss(output.mean(2), domain_sums)


class LA_CE(nn.Module):
    """
    Logit adjusted cross-entropy for long tail.
    """

    def __init__(self, config, dataset=None):
        super().__init__()
        class_counts = torch.tensor(dataset.get_class_counts())
        class_probs = class_counts / torch.sum(class_counts)
        tau = config.LA_tau
        epsilon = 1e-12

        self.adjustment = nn.parameter.Parameter(data=torch.log(torch.pow(class_probs, tau) + epsilon),
                                                 requires_grad=False)
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, output, target):
        logits = output
        la_logits = logits + self.adjustment
        return self.loss(la_logits, target[:, 0])


class GRL_A(nn.Module):
    """
    Standard cross entropy loss.
    """

    def __init__(self, config, dataset=None):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.domain = 1

    def __call__(self, output, target):
        logits = output
        return self.loss(logits, target[:, self.domain].long())

class GRL_L(nn.Module):
    """
    Standard cross entropy loss.
    """

    def __init__(self, config, dataset=None):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.domain = 2

    def __call__(self, output, target):
        logits = output
        return self.loss(logits, target[:, self.domain].long())

class BODA_A(nn.Module):
    """
    BODA on representations.
    """

    def __init__(self, config, dataset=None):
        super().__init__()
        self.config = config

        self.CE_loss = nn.CrossEntropyLoss()
        self.class_domain_counts = nn.parameter.Parameter(data=dataset.get_class_domain_counts(), requires_grad=False).cuda()
        self.use_c_d_mask = nn.parameter.Parameter(data=torch.where(self.class_domain_counts > 0, 1, 0),
                                                   requires_grad=False).cuda()
        # print("Class/domain exists in train dataset:")
        # print(self.use_c_d_mask)

        d_lambda = torch.zeros([config.n_classes, config.n_domains, config.n_classes, config.n_domains])

        # TODO: vectorise for release. Fine for now as is quick and only happens at init.
        for c1 in range(config.n_classes):
            for d1 in range(config.n_domains):
                for c2 in range(config.n_classes):
                    for d2 in range(config.n_domains):
                        d_lambda[c1, d1, c2, d2] = self.class_domain_counts[c1, d1] / (
                                self.class_domain_counts[c2, d2] + 1)
        self.d_lambda = nn.parameter.Parameter(data=torch.pow(d_lambda, self.config.boda_upsilon), requires_grad=False).cuda()

    def __call__(self, epoch, representation, c_d_means, target):
        """
        Input: dict containing representations for BODA, class statistics and epoch number
        target are the labels of shape batch x 2+, where dim 0 is the class, and dim 1 is the domain. Currently only uses class and 1 domain label
        """
        if epoch < self.config.stat_update_start_epoch:
            return torch.tensor(0.0, device=representation.device)

        l_class = target[:, 0]
        l_dom = target[:, 1]
        l_c_d = target[:, :2]
        reps = F.normalize(representation, dim=-1)
        means = F.normalize(c_d_means, dim=-1)
        num_d = len(torch.unique(l_dom))
        n_batch = reps.shape[0]
        rep_dim = reps.shape[-1]
        device = reps.device
        epsilon = 1e-8
        # means (60, 10, 4096)
        # reps (128, 4096)
        # dists (128, 60, 10, 4096)

        # calculate distance of every z to every c,d representation
        if self.config.boda_dist == "mean":
            dists = einops.rearrange(reps, 'b d -> b 1 1 d') - means
            dists = torch.norm(dists, dim=-1)
            # TODO: Divide by sqrt dim(representations)? Not in paper but might help with large loss as dim=4096 for MLP
            # dists = dists / math.sqrt(rep_dim)

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
        z_d_mask = torch.nn.functional.one_hot(l_dom.long(), self.config.n_domains).float()

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
        # TODO: better done with matmul?
        numerator = torch.mul(exp_dists, einops.repeat(z_c_mask, 'b c -> b c d', d=self.config.n_domains))
        numerator = torch.sum(numerator, dim=1)

        # denominators: z x d
        v = torch.ones(n_batch, device=device)
        z_idxs = torch.arange(n_batch, device=device).reshape(-1, 1)
        l_z_c_d = torch.cat([z_idxs, l_c_d], dim=-1)
        z_c_d_mask = torch.sparse_coo_tensor(l_z_c_d.t(), v,
                                             [n_batch, self.config.n_classes, self.config.n_domains]).to_dense()
        denominator = einops.repeat(exp_dists_sum, 'b -> b d', d=self.config.n_domains) - torch.sum(
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

        # should we divide boda_loss by batch size as well? The paper does not mention it, but it seems like good practice.
        boda_loss = boda_loss / n_batch

        return boda_loss

class BODA_L(nn.Module):
    """
    BODA on representations.
    """

    def __init__(self, config, dataset=None):
        super().__init__()
        self.config = config

        self.CE_loss = nn.CrossEntropyLoss()
        self.class_domain_counts = nn.parameter.Parameter(data=dataset.get_class_source_counts(), requires_grad=False).cuda()
        self.use_c_d_mask = nn.parameter.Parameter(data=torch.where(self.class_domain_counts > 0, 1, 0),
                                                   requires_grad=False).cuda()
        # print("Class/domain exists in train dataset:")
        # print(self.use_c_d_mask)

        d_lambda = torch.zeros([config.n_classes, config.n_sources, config.n_classes, config.n_sources])

        # TODO: vectorise for release. Fine for now as is quick and only happens at init.
        for c1 in range(config.n_classes):
            for d1 in range(config.n_sources):
                for c2 in range(config.n_classes):
                    for d2 in range(config.n_sources):
                        d_lambda[c1, d1, c2, d2] = self.class_domain_counts[c1, d1] / (
                                self.class_domain_counts[c2, d2] + 1)
        self.d_lambda = nn.parameter.Parameter(data=torch.pow(d_lambda, self.config.boda_upsilon), requires_grad=False).cuda()

    def __call__(self, epoch, representation, c_d_means, target):
        """
        Input: dict containing representations for BODA, class statistics and epoch number
        target are the labels of shape batch x 2+, where dim 0 is the class, and dim 1 is the domain. Currently only uses class and 1 domain label
        """
        if epoch < self.config.stat_update_start_epoch:
            return torch.tensor(0.0, device=representation.device)

        l_class = target[:, 0]
        l_dom = target[:, 2]
        l_c_d = target[:, [0,2]]
        reps = F.normalize(representation, dim=-1)
        means = F.normalize(c_d_means, dim=-1)
        num_d = len(torch.unique(l_dom))
        n_batch = reps.shape[0]
        rep_dim = reps.shape[-1]
        device = reps.device
        epsilon = 1e-8
        # means (60, 10, 4096)
        # reps (128, 4096)
        # dists (128, 60, 10, 4096)

        # calculate distance of every z to every c,d representation
        if self.config.boda_dist == "mean":
            dists = einops.rearrange(reps, 'b d -> b 1 1 d') - means
            dists = torch.norm(dists, dim=-1)
            # TODO: Divide by sqrt dim(representations)? Not in paper but might help with large loss as dim=4096 for MLP
            # dists = dists / math.sqrt(rep_dim)

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
        z_d_mask = torch.nn.functional.one_hot(l_dom.long(), self.config.n_sources).float()

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
        # TODO: better done with matmul?
        numerator = torch.mul(exp_dists, einops.repeat(z_c_mask, 'b c -> b c d', d=self.config.n_sources))
        numerator = torch.sum(numerator, dim=1)

        # denominators: z x d
        v = torch.ones(n_batch, device=device)
        z_idxs = torch.arange(n_batch, device=device).reshape(-1, 1)
        l_z_c_d = torch.cat([z_idxs, l_c_d], dim=-1)
        z_c_d_mask = torch.sparse_coo_tensor(l_z_c_d.t(), v,
                                             [n_batch, self.config.n_classes, self.config.n_sources]).to_dense()
        denominator = einops.repeat(exp_dists_sum, 'b -> b d', d=self.config.n_sources) - torch.sum(
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

        # should we divide boda_loss by batch size as well? The paper does not mention it, but it seems like good practice.
        boda_loss = boda_loss / n_batch

        return boda_loss

class AbstractMMD_A(nn.Module):
    """
    MMD and CORAL adpated from DomainBed.
    """

    def __init__(self, config, kernel, dataset=None):
        super().__init__()
        self.config = config
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

        l_dom = target[:, 1]
        reps = output.to(self.device)

        penalty = 0

        for i in range(self.config.n_domains):
            for j in range(i + 1, self.config.n_domains):

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

                penalty += self.mmd(x_i, x_j)

        if (self.config.n_domains) > 1 and not self.kernel_type == 'cov':
            penalty /= ((self.config.n_domains) * ((self.config.n_domains) - 1) / 2)

        loss = penalty
        # print(loss.item())

        return loss

class AbstractMMD_L(nn.Module):
    """
    MMD and CORAL adpated from DomainBed.
    """

    def __init__(self, config, kernel, dataset=None):
        super().__init__()
        self.config = config
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

        l_dom = target[:, 2]
        reps = output.to(self.device)

        penalty = 0

        for i in range(self.config.n_domains):
            for j in range(i + 1, self.config.n_domains):

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

                penalty += self.mmd(x_i, x_j)

        if (self.config.n_domains) > 1 and not self.kernel_type == 'cov':
            penalty /= ((self.config.n_domains) * ((self.config.n_domains) - 1) / 2)

        loss = penalty
        # print(loss.item())

        return loss

class MMD_gaussian_A(AbstractMMD_A):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, config, dataset=None):
        super(MMD_gaussian_A, self).__init__(config, kernel='gaussian', dataset=None)

class MMD_gaussian_L(AbstractMMD_L):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, config, dataset=None):
        super(MMD_gaussian_L, self).__init__(config, kernel='gaussian', dataset=None)

'''
class MMD_linear(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, config, dataset=None):
        super(MMD_linear, self).__init__(config, kernel='linear', dataset=None)
'''

class CORAL_A(AbstractMMD_A):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, config, dataset=None):
        super(CORAL_A, self).__init__(config, kernel='cov', dataset=None)

class CORAL_L(AbstractMMD_L):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, config, dataset=None):
        super(CORAL_L, self).__init__(config, kernel='cov', dataset=None)





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



