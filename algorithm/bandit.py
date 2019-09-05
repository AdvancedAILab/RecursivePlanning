
# bandit algorithms

import numpy as np

def mask_p(p, mask):
    p = (p + 1e-16) *mask
    return p / p.sum() 

def pucb(s, p):
    q_sum_all, n_all = s.q_sum_all + s.v / 2, s.n_all + 1
    q = (q_sum_all / n_all + s.q_sum) / (1 + s.n)
    pucb = q + 2.0 * np.sqrt(n_all) * p / (s.n + 1) - s.action_mask
    action = np.argmax(pucb)
    return action, pucb

def prepare_thompson(s, n_prior=1):
    q_sum_all, n_all = s.q_sum_all + s.v / 2 * n_prior, s.n_all + n_prior
    q_sum, n = s.q_sum + q_sum_all / n_all * n_prior, s.n + n_prior # for n + n_prior games
    q01_sum = (q_sum + n) / 2
    alpha, beta = q01_sum, n - q01_sum + s.action_mask
    return alpha, beta

def thompson(s, n_prior=1):
    alpha, beta = prepare_thompson(s, n_prior)
    r = np.random.beta(alpha, beta)
    return np.argmax(r), r

def pthompson(s, p):
    if s.n_all == 0:
        return np.random.choice(np.arange(len(p)), p=p), None
    p_mod =p / np.max(s.p)
    p_sum, ba = 0, None
    for _ in range(16):
        a, _ = thompson(s, 4)
        r = np.random.random(2)
        if r[0] < p_mod[a]:
            return a, None
        if r[1] * (p_sum + p_mod[a]) >= p_sum:
            ba = a
            p_sum += p_mod[a]
    return ba, None

def thompson_posterior(s, n_prior):
    alpha, beta = prepare_thompson(s, n_prior)
    posterior = np.zeros(len(s.p))
    samples = 10000
    for _ in range(samples):
        r = np.random.beta(alpha, beta)
        posterior[np.argmax(r)] += 1
    return posterior / samples
