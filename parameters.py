"""Build and return the parameter dictionary."""


def get_parameters(lx, ly, ka, gamma, eta, xi, lmin, delta_t, zeta, tau, q):
    return {
        "lx": lx,
        "ly": ly,
        "ka": ka,
        "gamma": gamma,
        "eta": eta,
        "xi": xi,
        "lmin": lmin,
        "delta_t": delta_t,
        "zeta": zeta,
        "tau": tau,
        "q": q,
    }