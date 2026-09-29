# ! -------------------------------------------- #
# ! ------------------- NOTE ------------------- #
# ! -------------------------------------------- #

# ! This file is taken verbatim from:
# ! https://github.com/shijiel2/TCD_v2 (mycode/certify_utils.py)
# ! See: https://arxiv.org/pdf/2308.07553


# ! -------------------------------------------- #
# ! ----------------- END NOTE ----------------- #
# ! -------------------------------------------- #

from __future__ import print_function

import math
from os import EX_OSFILE

import numpy as np
import scipy.stats
import torch
from autodp import utils
from loguru import logger
from opacus import PrivacyEngine
from opacus.accountants.analysis import rdp as privacy_analysis
from statsmodels.stats.proportion import proportion_confint

DEFAULT_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
DEFAULT_DELTA = 1e-05


def multi_ci(counts, alpha):
    multi_list = []
    n = np.sum(counts)
    l = len(counts)
    for i in range(l):
        multi_list.append(
            proportion_confint(
                # counts[i],
                min(max(counts[i], 1e-10), n - 1e-10),
                n,
                alpha=alpha / 2,
                method="beta",
            )
        )
    return np.array(multi_list)


def single_ci(counts, alpha):
    a = 1.0 * np.array(counts)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + (1 - alpha)) / 2.0, n - 1)
    return m, m - h, m + h, se * np.sqrt(n)


def multi_ci_bagging(counts, alpha):
    multi_list = []
    n = np.sum(counts)
    l = len(counts)
    for i in range(l):
        multi_list.append(proportion_confint(min(max(counts[i], 1e-10), n - 1e-10), n, alpha=alpha * 2.0 / l, method="beta"))
    return np.array(multi_list)


def multi_ci_softmax_hoeffding(dist, n_runs, alpha):
    interval = np.sqrt(1 / (2 * n_runs) * np.log((20) / alpha))
    upper = dist + interval
    lower = dist - interval
    return np.array(list(zip(lower, upper)))


def multi_ci_softmax_normal(dist, alpha):
    multi_list = []
    sigma_list = []
    for i in range(dist.shape[1]):
        single_dist = dist[:, i]
        m, l, u, sigma = single_ci(single_dist, alpha)
        multi_list.append([l, u])
        sigma_list.append(sigma)
    return np.array(multi_list), np.array(sigma_list)


def dp_amplify(epsilon, delta, m, n):

    mu = m / n
    delta_new = mu * delta
    epsilon_new = np.log(1 + mu * (np.e**epsilon - 1))

    return epsilon_new, delta_new


def rdp_amplify(alpha, m, n, sample_rate, sigma):

    prob = m / n

    print(f"m:{m}, n:{n}, prob:{prob}")

    def func(alpha):
        # ! The call below is deprecated
        # rdp = PrivacyEngine._get_renyi_divergence(sample_rate=sample_rate, noise_multiplier=sigma, alphas=[alpha])
        rdp = privacy_analysis.compute_rdp(q=sample_rate, noise_multiplier=sigma, steps=1, orders=[alpha])
        # eps = rdp.cpu().detach().numpy()[0]
        eps = rdp[0]
        return eps

    def cgf(x):
        return x * func(x + 1)

    def subsample_epsdelta(eps, delta, prob):
        if prob == 0:
            return 0, 0
        return np.log(1 + prob * (np.exp(eps) - 1)), prob * delta

    def subsample_func_int(x):
        # output the cgf of the subsampled mechanism
        mm = int(x)
        eps_inf = func(np.inf)

        moments_two = (
            2 * np.log(prob)
            + utils.logcomb(mm, 2)
            + np.minimum(
                np.log(4) + func(2.0) + np.log(1 - np.exp(-func(2.0))),
                func(2.0) + np.minimum(np.log(2), 2 * (eps_inf + np.log(1 - np.exp(-eps_inf)))),
            )
        )

        def moment_bound(j):
            return np.minimum(j * (eps_inf + np.log(1 - np.exp(-eps_inf))), np.log(2)) + cgf(j - 1) + j * np.log(prob) + utils.logcomb(mm, j)

        moments = [moment_bound(j) for j in range(3, mm + 1, 1)]
        return np.minimum((x - 1) * func(x), utils.stable_logsumexp([0, moments_two] + moments))

    def subsample_func(x):
        # This function returns the RDP at alpha = x
        # RDP with the linear interpolation upper bound of the CGF

        epsinf, tmp = subsample_epsdelta(func(np.inf), 0, prob)

        if np.isinf(x):
            return epsinf
        if prob == 1.0:
            return func(x)

        if (x >= 1.0) and (x <= 2.0):
            return np.minimum(epsinf, subsample_func_int(2.0) / (2.0 - 1))
        if np.equal(np.mod(x, 1), 0):
            return np.minimum(epsinf, subsample_func_int(x) / (x - 1))
        xc = math.ceil(x)
        xf = math.floor(x)
        return np.min([epsinf, func(x), ((x - xf) * subsample_func_int(xc) + (1 - (x - xf)) * subsample_func_int(xf)) / (x - 1)])

    return alpha, subsample_func(alpha)


def check_condition_dp(args, radius_value, epsilon, delta, p_l_value, p_s_value):
    if radius_value == 0:
        return True

    r, e, d, pl, ps = np.float64(radius_value), np.float64(epsilon), np.float64(delta), np.float64(p_l_value), np.float64(p_s_value)

    if args.train_mode == "Sub-DP":
        e, d = dp_amplify(e, d, args.sub_training_size, args.training_size)

    group_eps = e * r
    group_delta = d * r

    upper = (np.e ** (group_eps)) * ps + group_delta
    lower = (np.e ** (-group_eps)) * (pl - group_delta)

    # print(r, e, d, pl, ps)
    try:
        val = lower - upper
    except Exception:
        val = 0

    if val > 0:
        return True
    else:
        return False


def check_condition_dp_baseline(args, radius_value, epsilon, delta, p_l_value, p_s_value):
    if radius_value == 0:
        return True

    r, e, d, pl, ps = np.float64(radius_value), np.float64(epsilon), np.float64(delta), np.float64(p_l_value), np.float64(p_s_value)

    lower = np.e ** (-r * e) * (pl + d / (np.e**e - 1))
    upper = np.e ** (r * e) * (ps + d / (np.e**e - 1))

    # print(r, e, d, pl, ps)
    try:
        val = lower - upper
    except Exception:
        val = 0

    if val > 0:
        return True
    else:
        return False


def rdp_bounds(
    radius,
    sample_rate,
    steps,
    alpha,
    sigma,
    p1,
    p2,
    softmax,
    agg_res_param=None,
    amplify=False,
    training_size=None,
    sub_training_size=None,
    clip=True,
):
    sample_rate_gp = 1 - (1 - sample_rate) ** radius
    if not amplify:
        # rdp = PrivacyEngine._get_renyi_divergence(sample_rate=sample_rate_gp, noise_multiplier=sigma, alphas=[alpha]) * steps
        # eps = rdp.cpu().detach().numpy()[0]
        rdp = privacy_analysis.compute_rdp(q=sample_rate_gp, noise_multiplier=sigma, steps=steps, orders=[alpha])
        eps = rdp[0]
    else:
        _, eps = rdp_amplify(alpha, sub_training_size, training_size, sample_rate_gp, sigma)
        eps *= steps

    if not softmax:
        lower = np.e ** (-eps) * p1 ** (alpha / (alpha - 1))
        upper = (np.e**eps * p2) ** ((alpha - 1) / alpha)
    else:
        if agg_res_param is not None:
            mgf_diff_list, varis_p1, varis_p2 = agg_res_param["mgf_diff_list"], agg_res_param["varis_p1_p2"][0], agg_res_param["varis_p1_p2"][1]

            def moments(order, mean, varis):
                from sympy import symbols

                t, mu, sigma = symbols("t, mu, sigma")
                return mgf_diff_list[order].evalf(subs={t: 0, mu: mean, sigma: varis})

            a1a = (alpha - 1) / alpha
            aa1 = round(alpha / (alpha - 1))
            upper_moments = moments(aa1, p2, varis_p2)
            upper = np.e ** (eps * a1a) * upper_moments**a1a
            lower = np.e ** (-eps) * p1**aa1

            # def moments_bounds(agg_res, order):
            #     agg_res_pow = np.power(agg_res, order)
            #     return single_ci(agg_res_pow, 1e-5)

            # p1_agg_res = agg_res[:, p1_idx]
            # p2_agg_res = agg_res[:, p2_idx]
            # _, l1, _ = moments_bounds(p1_agg_res, (alpha-1)/alpha)
            # _, _, u2 = moments_bounds(p2_agg_res, alpha/(alpha-1))

            # if l1 == 1 or u2 == 0:
            #     p1_agg_res = np.longdouble(agg_res[:, p1_idx])
            #     p2_agg_res = np.longdouble(agg_res[:, p2_idx])
            #     _, l1, _ = moments_bounds(p1_agg_res, (alpha-1)/alpha)
            #     _, _, u2 = moments_bounds(p2_agg_res, alpha/(alpha-1))

            # lower = np.e**(-eps) * l1**(alpha/(alpha-1))
            # upper = (np.e**eps * u2)**((alpha-1)/alpha)
        else:
            lower = np.e ** (-eps) * p1 ** (alpha / (alpha - 1))
            upper = (np.e**eps * p2) ** ((alpha - 1) / alpha)

    if clip:
        return min(max(lower, 0), 1), max(min(upper, 1), 0)
    else:
        return lower, upper


def check_condition_rdp_deprecated(args, radius, sample_rate, steps, sigma, p1, p2, softmax, agg_res_param=None):

    if radius == 0:
        return True

    max_lower = 0
    min_upper = 1
    for alpha in [1 + x / 10 for x in range(1, 100)]:
        if args.train_mode == "DP" or args.train_mode == "Sub-DP-no-amp":
            lower, upper = rdp_bounds(radius, sample_rate, steps, alpha, sigma, p1, p2, softmax, agg_res_param=agg_res_param)
        elif args.train_mode == "Sub-DP":
            lower, upper = rdp_bounds(
                radius,
                sample_rate,
                steps,
                alpha,
                sigma,
                p1,
                p2,
                softmax,
                agg_res_param=agg_res_param,
                amplify=True,
                training_size=args.training_size,
                sub_training_size=args.sub_training_size,
            )
        if lower > max_lower:
            max_lower = lower
        if upper < min_upper:
            min_upper = upper
    val = max_lower - min_upper
    if val > 0:
        return True
    else:
        return False


def is_integer(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()


def check_condition_rdp(args, radius, sample_rate, steps, sigma, p1, p2, softmax, agg_res_param=None):
    if radius == 0:
        return True

    # print(args.train_mode)

    def bounds(alpha):
        if args.train_mode == "DP" or args.train_mode == "Sub-DP-no-amp":
            lower, upper = rdp_bounds(radius, sample_rate, steps, alpha, sigma, p1, p2, softmax, agg_res_param=agg_res_param)
        elif args.train_mode == "Sub-DP":
            lower, upper = rdp_bounds(
                radius,
                sample_rate,
                steps,
                alpha,
                sigma,
                p1,
                p2,
                softmax,
                agg_res_param=agg_res_param,
                amplify=True,
                training_size=args.training_size,
                sub_training_size=args.sub_training_size,
            )
        return lower, upper

    # alphas, uppers, lowers = [], [], []

    if agg_res_param is None:
        alpha_range = [1 + x / 100 for x in range(1, 1001)]
    else:
        # need to make sure alpha/(alpha-1) is integer
        aa1_range = list(range(2, 1002))
        alpha_range = [x / (x - 1) for x in aa1_range].sort()

    max_lower, min_upper = bounds(alpha_range[0])
    upper_stop, lower_stop = False, False
    for alpha in alpha_range[1:]:
        lower, upper = bounds(alpha)
        # stop condition for upper and lower bounds, as they may occur in different alpha
        if lower <= max_lower:
            lower_stop = True
        if upper >= min_upper:
            upper_stop = True
            # if min_upper is 1, no matter what max_lower is the val should always < 1.
            if min_upper == 1:
                break
        if lower_stop and upper_stop:
            break
        # update min/max if we find better bounds
        if lower > max_lower:
            max_lower = lower
        if upper < min_upper:
            min_upper = upper
        # we find the one need
        if max_lower - min_upper > 0:
            break

        # alphas.append(alpha)
        # uppers.append(upper)
        # lowers.append(lower)
    # import matplotlib.pyplot as plt
    # plt.plot(alphas, list(map(lambda x: x*100, lowers)))
    # plt.savefig("alpha_bounds_test1.png", bbox_inches='tight')
    # plt.clf()
    # plt.plot(alphas, list(map(lambda x: x*100, uppers)))
    # plt.savefig("alpha_bounds_test2.png", bbox_inches='tight')
    # plt.clf()

    val = max_lower - min_upper
    if val > 0:
        return True
    else:
        return False


def check_condition_rdp_gp(args, radius, sample_rate, steps, alpha, delta, sigma, p1, p2):

    if radius == 0:
        return True

    if args.train_mode == "DP" or args.train_mode == "Sub-DP-no-amp":
        # rdp = PrivacyEngine._get_renyi_divergence(sample_rate=sample_rate, noise_multiplier=sigma, alphas=[alpha]) * steps
        # eps = rdp.cpu().detach().numpy()[0]
        rdp = privacy_analysis.compute_rdp(q=sample_rate, noise_multiplier=sigma, steps=steps, orders=[alpha])
        eps = rdp[0]
    elif args.train_mode == "Sub-DP":
        _, eps = rdp_amplify(alpha, args.sub_training_size, args.training_size, sample_rate, sigma)
        eps *= steps

    alpha = alpha / radius
    eps = 3 ** (np.log2(radius)) * eps

    if alpha <= 1:
        return False

    val = np.e ** (-eps) * p1 ** (alpha / (alpha - 1)) - (np.e**eps * p2) ** ((alpha - 1) / alpha)
    if val > 0:
        return True
    else:
        return False


def check_condition_dp_bagging(radius_value, k_value, n_value, p_l_value, p_s_value, dp_rad):

    if radius_value == 0:
        return True

    import math

    def nCr(n, r):
        f = math.factorial
        return int(f(n) / f(r) / f(n - r))

    def binoSum(k, x_start, p):
        prob_sum = 0
        for x in range(x_start, k + 1):
            prob = nCr(k, x) * p**x * (1 - p) ** (k - x)
            prob_sum += prob
        return prob_sum

    p3 = binoSum(k_value, (k_value - dp_rad), (n_value - radius_value) / n_value)
    lower = p_l_value - (1 - p3)
    upper = p_s_value + (1 - p3)

    try:
        val = lower - upper
    except Exception:
        val = 0

    if val > 0:
        return True
    else:
        return False


def check_condition_dp_bagging_softmax_prob(radius_value, sample_rate, steps, alpha, sigma, k_value, n_value, l1_lower, l1_upper, l2_lower, l2_upper):
    if radius_value == 0:
        return True

    pi_l1_lower, pi_l1_upper = rdp_bounds(1, sample_rate, steps, alpha, sigma, l1_lower, l1_upper, softmax=True)
    pi_l2_lower, pi_l2_upper = rdp_bounds(1, sample_rate, steps, alpha, sigma, l2_lower, l2_upper, softmax=True)

    p = 1 - ((n_value - radius_value) / n_value) ** k_value
    omega1 = (pi_l1_upper - pi_l1_lower) * p
    omega2 = (pi_l2_upper - pi_l2_lower) * p

    val1 = l1_lower - omega1
    val2 = l2_upper + omega2
    val = val1 - val2

    if val > 0:
        return True
    else:
        return False


def check_condition_bagging(radius_value, k_value, n_value, p_l_value, p_s_value):

    threshold_point = radius_value / (1.0 - np.power(0.5, 1.0 / (k_value - 1.0)))

    if threshold_point <= n_value:
        nprime_value = int(n_value)
        value_check = compute_compare_value_bagging(radius_value, nprime_value, k_value, n_value, p_l_value, p_s_value)
    elif threshold_point >= n_value + radius_value:
        nprime_value = int(n_value + radius_value)
        value_check = compute_compare_value_bagging(radius_value, nprime_value, k_value, n_value, p_l_value, p_s_value)
    else:
        nprime_value_1 = np.ceil(threshold_point)
        value_check_1 = compute_compare_value_bagging(radius_value, nprime_value_1, k_value, n_value, p_l_value, p_s_value)
        nprime_value_2 = np.floor(threshold_point)
        value_check_2 = compute_compare_value_bagging(radius_value, nprime_value_2, k_value, n_value, p_l_value, p_s_value)
        value_check = max(value_check_1, value_check_2)
    if value_check < 0:
        return True
    else:
        return False


def compute_compare_value_bagging(radius_cmp, nprime_cmp, k_cmp, n_cmp, p_l_cmp, p_s_cmp):
    return (
        np.power(float(nprime_cmp) / float(n_cmp), k_cmp)
        - 2 * np.power((float(nprime_cmp) - float(radius_cmp)) / float(n_cmp), k_cmp)
        + 1
        - p_l_cmp
        + p_s_cmp
    )


def CertifyRadiusDP(args, ls, CI, epsilon, delta):
    radius = 0
    p_ls, runner_up_prob = top2_probs(CI, ls)
    if p_ls <= runner_up_prob:
        return -1
    # this is where to calculate the r
    low, high = 0, 200
    while low <= high:
        radius = math.ceil((low + high) / 2.0)
        if check_condition_dp(args, radius, epsilon, delta, p_ls, runner_up_prob):
            low = radius + 0.1
        else:
            high = radius - 1
    radius = math.floor(low)
    if check_condition_dp(args, radius, epsilon, delta, p_ls, runner_up_prob):
        return radius
    else:
        print("error")
        raise ValueError


def CertifyRadiusDP_baseline(args, ls, CI, epsilon, delta):
    radius = 0
    p_ls, runner_up_prob = top2_probs(CI, ls)
    if p_ls <= runner_up_prob:
        return -1
    # this is where to calculate the r
    low, high = 0, 200
    while low <= high:
        radius = math.ceil((low + high) / 2.0)
        if check_condition_dp_baseline(args, radius, epsilon, delta, p_ls, runner_up_prob):
            low = radius + 0.1
        else:
            high = radius - 1
    radius = math.floor(low)
    if check_condition_dp_baseline(args, radius, epsilon, delta, p_ls, runner_up_prob):
        return radius
    else:
        print("error")
        raise ValueError


def CertifyRadiusRDP(args, ls, CI, steps, sample_rate, sigma, softmax=False):
    p1, p2 = top2_probs(CI, ls)
    if p1 <= p2:
        return -1

    valid_radius = set()
    # binary search for radius
    low, high = 0, 200
    while low <= high:
        radius = math.ceil((low + high) / 2.0)
        if check_condition_rdp(args, radius=radius, sample_rate=sample_rate, steps=steps, sigma=sigma, p1=p1, p2=p2, softmax=softmax):
            low = radius + 0.1
        else:
            high = radius - 1
    radius = math.floor(low)
    if check_condition_rdp(args, radius=radius, sample_rate=sample_rate, steps=steps, sigma=sigma, p1=p1, p2=p2, softmax=softmax):
        valid_radius.add(radius)
    elif radius == 0:
        valid_radius.add(radius)
    else:
        print("error", radius)
        raise ValueError

    if len(valid_radius) > 0:
        max_radius = max(valid_radius)
        # for x in valid_radius:
        #     if x[0] == max_radius:
        #         print(x)
        return max_radius
    else:
        return 0


def CertifyRadiusRDP_moments(args, ls, CI, steps, sample_rate, sigma, mgf_diff_list, varis, softmax=False):
    p1, p2 = top2_probs(CI, ls)
    p1_idx, p2_idx = top2_probs(CI, ls, return_index=True)
    varis_p1_p2 = (varis[p1_idx], varis[p2_idx])
    agg_res_param = {"p1_idx": p1_idx, "p2_idx": p2_idx, "mgf_diff_list": mgf_diff_list, "varis_p1_p2": varis_p1_p2}
    if p1 <= p2:
        return -1

    valid_radius = set()
    # binary search for radius
    low, high = 0, 200
    while low <= high:
        radius = math.ceil((low + high) / 2.0)
        if check_condition_rdp(
            args, radius=radius, sample_rate=sample_rate, steps=steps, sigma=sigma, p1=p1, p2=p2, softmax=softmax, agg_res_param=agg_res_param
        ):
            low = radius + 0.1
        else:
            high = radius - 1
    radius = math.floor(low)
    if check_condition_rdp(
        args, radius=radius, sample_rate=sample_rate, steps=steps, sigma=sigma, p1=p1, p2=p2, softmax=softmax, agg_res_param=agg_res_param
    ):
        valid_radius.add(radius)
    elif radius == 0:
        valid_radius.add(radius)
    else:
        print("error", radius)
        raise ValueError

    if len(valid_radius) > 0:
        max_radius = max(valid_radius)
        # for x in valid_radius:
        #     if x[0] == max_radius:
        #         print(x)
        return max_radius
    else:
        return 0


def CertifyRadiusRDP_GP(args, ls, CI, steps, sample_rate, sigma):
    p1, p2 = top2_probs(CI, ls)
    if p1 <= p2:
        return -1

    valid_radius = set()
    for alpha in [1 + x for x in range(1, 100)]:
        # for delta in [x / 100.0 for x in range(1, 10)]:
        for delta in [0]:
            # binary search for radius
            low, high = 0, 200
            while low <= high:
                radius = math.ceil((low + high) / 2.0)
                if check_condition_rdp_gp(
                    args, radius=radius, sample_rate=sample_rate, steps=steps, alpha=alpha, delta=delta, sigma=sigma, p1=p1, p2=p2
                ):
                    low = radius + 0.1
                else:
                    high = radius - 1
            radius = math.floor(low)
            if check_condition_rdp_gp(args, radius=radius, sample_rate=sample_rate, steps=steps, alpha=alpha, delta=delta, sigma=sigma, p1=p1, p2=p2):
                valid_radius.add((radius, alpha, delta))
            elif radius == 0:
                valid_radius.add((radius, alpha, delta))
            else:
                print("error", (radius, alpha, delta))
                raise ValueError

    if len(valid_radius) > 0:
        max_radius = max(valid_radius, key=lambda x: x[0])[0]
        # for x in valid_radius:
        #     if x[0] == max_radius:
        #         print(x)
        return max_radius
    else:
        return 0


def CertifyRadiusBS(ls, CI, k, n):
    radius = 0
    p_ls, runner_up_prob = top2_probs(CI, ls)
    if p_ls <= runner_up_prob:
        return -1
    low, high = 0, 200
    while low <= high:
        radius = math.ceil((low + high) / 2.0)
        if check_condition_bagging(radius, k, n, p_ls, runner_up_prob):
            low = radius + 0.1
        else:
            high = radius - 1
    radius = math.floor(low)
    if check_condition_bagging(radius, k, n, p_ls, runner_up_prob):
        return radius
    else:
        print("error")
        raise ValueError


def CertifyRadiusDPBS(args, ls, CI, k, n, epsilon, delta, steps, sample_rate, sigma, softmax=False):
    # first using CertifyRadius_DP to find out the robustness we have in a sub-dataset
    # change train_mode to 'DP' to avoid dp amplification
    args.train_mode = "DP"
    dp_rad = max(0, CertifyRadiusRDP(args, ls, CI, steps, sample_rate, sigma, softmax=softmax), CertifyRadiusDP(args, ls, CI, epsilon, delta))
    args.train_mode = "Sub-DP"

    # DP bagging part
    radius = 0
    p_ls, runner_up_prob = top2_probs(CI, ls)
    if p_ls <= runner_up_prob:
        return -1, dp_rad
    low, high = 0, 200
    while low <= high:
        radius = math.ceil((low + high) / 2.0)
        if check_condition_dp_bagging(radius, k, n, p_ls, runner_up_prob, dp_rad):
            low = radius + 0.1
        else:
            high = radius - 1
    radius = math.floor(low)
    if check_condition_dp_bagging(radius, k, n, p_ls, runner_up_prob, dp_rad):
        return radius, dp_rad
    else:
        print("error")
        raise ValueError


def CertifyRadiusDPBS_softmax_prob(ls, CI, k, n, delta, steps, sample_rate, sigma, upper=1000, ignore_label=False):
    l1_idx, l2_idx = top2_probs(CI, ls, return_index=True)
    l1_lower, l1_upper = CI[l1_idx][0], CI[l1_idx][1]
    l2_lower, l2_upper = CI[l2_idx][0], CI[l2_idx][1]

    if l1_lower <= l2_upper and not ignore_label:
        return -1

    valid_radius = set()
    for alpha in DEFAULT_ALPHAS:  # [1 + x / 10 for x in range(1, 1000)]:
        # for delta in [x / 100.0 for x in range(1, 10)]:
        for delta in [0]:
            # binary search for radius
            low, high = 0, upper
            while low <= high:
                radius = math.ceil((low + high) / 2.0)
                if check_condition_dp_bagging_softmax_prob(radius, sample_rate, steps, alpha, sigma, k, n, l1_lower, l1_upper, l2_lower, l2_upper):
                    low = radius + 0.1
                else:
                    high = radius - 1
            radius = math.floor(low)
            if check_condition_dp_bagging_softmax_prob(radius, sample_rate, steps, alpha, sigma, k, n, l1_lower, l1_upper, l2_lower, l2_upper):
                valid_radius.add((radius, alpha, delta))
            elif radius == 0:
                valid_radius.add((radius, alpha, delta))
            else:
                print("error", (radius, alpha, delta))
                raise ValueError

    if len(valid_radius) > 0:
        max_radius = max(valid_radius, key=lambda x: x[0])[0]
        # for x in valid_radius:
        #     if x[0] == max_radius:
        #         print(x)
        return max_radius
    else:
        return 0


def get_rdp(sample_rate, noise_multiplier, num_steps, alphas=DEFAULT_ALPHAS):
    rdp = privacy_analysis.compute_rdp(
        q=sample_rate,
        noise_multiplier=noise_multiplier,
        steps=num_steps,
        orders=alphas,
    )
    return rdp


def get_cdp(sample_rate, noise_multiplier, num_steps, alphas=DEFAULT_ALPHAS):
    rdp = get_rdp(sample_rate, noise_multiplier, num_steps, alphas)
    eps, best_alpha = privacy_analysis.get_privacy_spent(orders=alphas, rdp=rdp, delta=DEFAULT_DELTA)
    return eps, best_alpha


def get_dir(train_mode, results_folder, model_name, lr, sigma, max_per_sample_grad_norm, sample_rate, epochs, n_runs, sub_training_size):
    max_per_sample_grad_norm = float(max_per_sample_grad_norm)
    if train_mode == "DP":
        result_folder = f"{results_folder}/{train_mode}_{model_name}_{lr}_{sigma}_" f"{max_per_sample_grad_norm}_{sample_rate}_{epochs}_{n_runs}"
    elif train_mode == "Bagging":
        result_folder = f"{results_folder}/{train_mode}_{model_name}_{lr}_{sub_training_size}_" f"{epochs}_{n_runs}"
    elif train_mode == "Sub-DP":
        result_folder = (
            f"{results_folder}/{train_mode}_{model_name}_{lr}_{sigma}_"
            f"{max_per_sample_grad_norm}_{sample_rate}_{epochs}_{sub_training_size}_{n_runs}"
        )
    elif train_mode == "Sub-DP-no-amp":
        result_folder = (
            f"{results_folder}/{train_mode}_{model_name}_{lr}_{sigma}_"
            f"{max_per_sample_grad_norm}_{sample_rate}_{epochs}_{sub_training_size}_{n_runs}"
        )
    else:
        exit("Invalid Method name.")
    print(result_folder)
    return result_folder


def extract_summary(dir_path):
    acc_list = np.load(f"{dir_path}/acc_list.npy")
    acc_avg = np.mean(acc_list)

    rdp_history = np.load(f"{dir_path}/rdp_history.npy")

    if len(rdp_history) > 1:
        raise RuntimeError("RDP history is longer than 1, why is that?")

    (noise_multiplier, sample_rate, num_steps) = rdp_history[0]
    eps, best_alpha = get_cdp(sample_rate, noise_multiplier, num_steps)

    return acc_avg, eps


def aggres_meta_info(aggregate_result):
    pred_data = aggregate_result[:, :10]
    pred = np.argmax(pred_data, axis=1)
    gt = aggregate_result[:, 10]
    return gt, pred


def confident_interval_multinomial(aggregate_result, idx, method_name, alpha):
    ls = aggregate_result[idx][-1]
    class_freq = aggregate_result[idx][:-1]
    if method_name == "bagging":
        CI = multi_ci_bagging(class_freq, alpha)
    else:
        CI = multi_ci(class_freq, alpha)
    return CI, ls


def confident_interval_softmax(aggregate_result_softmax, aggregate_result_softmax_rm, idx, method_name, alpha):
    if "softmax" in method_name:
        ls = int(aggregate_result_softmax_rm[idx][-1])
        # CI = multi_ci_softmax_hoeffding(aggregate_result_softmax_rm[idx][:-1], aggregate_result_softmax.shape[0], alpha)
        CI, sigmas = multi_ci_softmax_normal(aggregate_result_softmax[:, idx, :-1], alpha)
    return CI, ls, sigmas


def top2_probs(CI, ls, return_index=False):
    delta_l, delta_s = (
        1e-50,
        1e-50,
    )
    pABar = CI[ls][0]
    probability_bar = CI[:, 1] + delta_s
    probability_bar = np.clip(probability_bar, a_min=-1, a_max=1 - pABar)
    probability_bar[ls] = pABar - delta_l

    p_ls = probability_bar[ls]
    probability_bar[ls] = -1
    runner_up_prob = np.amax(probability_bar)
    if not return_index:
        return p_ls, runner_up_prob
    else:
        return ls, np.argmax(probability_bar)


def result_folder_path_generator(args):
    if args.train_mode == "DP":
        result_folder = (
            f"{args.results_folder}/{args.train_mode}_{args.model_name}_{args.lr}_{args.sigma}_"
            f"{args.max_per_sample_grad_norm}_{args.sample_rate}_{args.epochs}_{args.n_runs}"
        )
    elif args.train_mode == "Bagging":
        result_folder = (
            f"{args.results_folder}/{args.train_mode}_{args.model_name}_{args.lr}_{args.sub_training_size}_" f"{args.epochs}_{args.n_runs}"
        )
    elif args.train_mode == "Sub-DP":
        result_folder = (
            f"{args.results_folder}/{args.train_mode}_{args.model_name}_{args.lr}_{args.sigma}_"
            f"{args.max_per_sample_grad_norm}_{args.sample_rate}_{args.epochs}_{args.sub_training_size}_{args.n_runs}"
        )
    elif args.train_mode == "Sub-DP-no-amp":
        result_folder = (
            f"{args.results_folder}/{args.train_mode}_{args.model_name}_{args.lr}_{args.sigma}_"
            f"{args.max_per_sample_grad_norm}_{args.sample_rate}_{args.epochs}_{args.sub_training_size}_{args.n_runs}"
        )
    return result_folder


def gen_sub_dataset(dataset, sub_training_size, with_replacement):
    indexs = np.random.choice(len(dataset), sub_training_size, replace=with_replacement)
    dataset = torch.utils.data.Subset(dataset, indexs)
    return dataset


def freeze_params(parames_list):
    for p in parames_list:
        p.requires_grad = False


def unfreeze_params(parames_list):
    for p in parames_list:
        p.requires_grad = True
