import mpmath as mp
from mpmath import exp, pi, quad, sqrt


# Standard normal CDF
def Phi(x):
    return 0.5 * (1 + mp.erf(x / mp.sqrt(2)))


# δ(ε) as in the corollary
def delta_eps(eps, mu):
    return Phi(-eps / mu + mu / 2) - mp.e ** (eps) * Phi(-eps / mu - mu / 2)


# Solve for ε given mu and δ_target
def solve_epsilon(mu, delta_target):
    f = lambda eps: delta_eps(eps, mu) - delta_target
    return mp.findroot(f, [-500, 500], solver="bisect")  # initial guess can be tuned


# Example usage
sigmas = [0.1, 0.5, 2, 5]
for sigma in sigmas:
    mu = 1 / sigma
    delta_target = 1e-6
    epsilon = solve_epsilon(mu, delta_target)

    print(f"Sigma:{sigma} -> Epsilon:{epsilon}")
