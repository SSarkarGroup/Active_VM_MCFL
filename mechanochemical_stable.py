# code to incorporate the mechanochemical model and solve the ODEs.
import numpy as np
from scipy import optimize


b = 3
c = 1


def brusselator_w_mcf9(A, P, y, a, b, c, d0, alpha, beta, tauD, tauA, tauP, E0, q, A0hat, Ahat, Gamma, pf):
    """
    MCFL with ERK influencing A0, P0, and Gamma.
    """
    M, E, D, A0, P0, Gam = y

    if A0 <= 0:
        print("A0 not well-defined. A0:", A0)

    dMdt = a - (b + 1) * M + c * M**2 * E
    dEdt = b * M - c * M**2 * E - D * E
    d = D - d0
    hc = 2
    Sigma = A0 - 2 * 1 / (1 + (A / Ahat) ** hc)

    G = Gam - Gamma
    e = E - E0

    dddt = -(d + d**3) / tauD - beta * D * (A - 1) / tauD
    dSigmadt = -(Sigma + Sigma**3) / tauA - alpha * (e) / tauA
    dGdt = 0
    gammaP = 2 * q
    Lambda = P0 - gammaP * ((A / Ahat) ** hc / (1 + (A / Ahat) ** hc))
    dLambdadt = -(Lambda + Lambda**3) / tauP - 1 * (alpha / (pf * np.sqrt(A0))) * (e) / tauP
    return [dMdt, dEdt, dddt, dSigmadt, dLambdadt, dGdt]


def f(x, a, b, c, D0):
    M, E = x
    y1 = a - (b + 1) * M + c * E * M**2
    y2 = b * M - c * E * M**2 - D0 * E
    return [y1, y2]


def J(x, a, b, c, D0):
    M, E = x
    J11 = -(b + 1) + 2 * c * E * M
    J12 = c * M**2
    J21 = b - 2 * c * E * M
    J22 = -c * M**2 - D0
    return [[J11, J12], [J21, J22]]


def init_chem(polys):
    for poly in polys:
        d0 = poly.d0
        a = poly.a
        guess = np.array([0, 0])
        sol = optimize.root(f, guess, jac=J, method="hybr", args=(a, b, c, d0))
        poly.E0 = sol.x[1]
        poly.E = poly.E0
        poly.M = sol.x[0]


def update_chem(polys, vertices, parameters, L):
    delta_t = parameters["delta_t"]
    tauD = parameters["tauD"]
    tauA = parameters["tauA"]
    tauP = parameters["tauP"]
    Gamma = parameters["gamma"]
    q = parameters["q"]
    A0hat = parameters["A0hat"]
    Ahat = parameters["Ahat"]
    pf = parameters["pf"]

    for poly in polys:
        E = poly.E
        E0 = poly.E0
        M = poly.M
        D = poly.D
        A0 = poly.A0
        P0 = poly.P0
        d0 = poly.d0
        Gam = poly.Gam

        alpha = poly.alpha
        beta = poly.beta

        a = poly.a
        A = poly.get_area(vertices, L)
        P = poly.get_perim(vertices, L)

        y = [M, E, D, A0, P0, Gam]

        dMdt, dEdt, dddt, dSigmadt, dLambdadt, dGdt = brusselator_w_mcf9(
            A,
            P,
            y,
            a,
            b,
            c,
            d0,
            alpha,
            beta,
            tauD,
            tauA,
            tauP,
            E0,
            q,
            A0hat,
            Ahat,
            Gamma,
            pf,
        )

        poly.M += dMdt * delta_t
        poly.E += dEdt * delta_t
        poly.D += dddt * delta_t
        poly.A0 = poly.A0 + dSigmadt * delta_t
        poly.P0 = poly.P0 + dLambdadt * delta_t
        poly.Gam = poly.Gam + dGdt * delta_t

    return polys


def update_cyclin(polys, parameters, delta_t):
    for poly in polys:
        dcdt = 1
        poly.c += dcdt * delta_t

    return polys