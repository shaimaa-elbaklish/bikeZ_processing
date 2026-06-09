"""
TITLE OF PAPAER
-------------------------------------------
Authors:        Shaimaa El-Baklish
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2025
Submitted to:   JOURNAL
-------------------------------------------
"""

# #############################################################################
# IMPORTS
# #############################################################################
import gc
import sys
import warnings
warnings.simplefilter('ignore', RuntimeWarning) # Ignore all RuntimeWarnings

import numpy as np
import pandas as pd

from math import comb
from scipy.optimize import minimize
from pyclothoids import SolveG2, Clothoid


# #############################################################################
# FUNCTIONS
# #############################################################################
def bernstein_basis(degree, t):
    """
    Return (len(t), degree+1) Bernstein basis matrix evaluated at t in [0,1].
    """
    t = np.asarray(t)
    B = np.zeros((len(t), degree + 1))
    for i in range(degree + 1):
        B[:, i] = comb(degree, i) * (t ** i) * ((1 - t) ** (degree - i))
    return B


def bernstein_derivatives(degree, t_norm):
    """
    Returns B, B1, B2, B3: matrices of shape (N_times, degree+1)
    B  = Bernstein basis
    B1 = first derivative
    B2 = second derivative
    B3 = third derivative
    """
    from scipy.special import comb
    N = len(t_norm)
    n = degree
    B  = np.zeros((N, n+1))
    B1 = np.zeros((N, n+1))
    B2 = np.zeros((N, n+1))
    B3 = np.zeros((N, n+1))
    for i in range(n+1):
        B[:, i]  = comb(n, i) * t_norm**i * (1-t_norm)**(n-i)
    # First derivative
    for i in range(n+1):
        B1[:, i] = 0.0
        if i > 0:
            B1[:, i] += n * B[:, i-1]
        if i < n:
            B1[:, i] -= n * B[:, i]
    # Second derivative
    for i in range(n+1):
        B2[:, i] = 0.0
        if i > 1:
            B2[:, i] += n*(n-1) * B[:, i-2]
        if 0 < i < n:
            B2[:, i] -= 2*n*(n-1) * B[:, i-1]
        if i < n-1:
            B2[:, i] += n*(n-1) * B[:, i]
    # Third derivative
    for i in range(n+1):
        B3[:, i] = 0.0
        if i > 2:
            B3[:, i] += n*(n-1)*(n-2) * B[:, i-3]
        if 1 < i < n:
            B3[:, i] -= 3*n*(n-1)*(n-2) * B[:, i-2]
        if 0 < i < n-1:
            B3[:, i] += 3*n*(n-1)*(n-2) * B[:, i-1]
        if i < n-2:
            B3[:, i] -= n*(n-1)*(n-2) * B[:, i]
    return B, B1, B2, B3


def _monotone_alpha_from_beta(beta):
    """
    Given beta (R^n), returns monotone nondecreasing alpha via cumulative sum of exp.
    alpha_i = sum_{k=0..i} exp(beta_k)
    """
    e = np.exp(beta)
    return np.cumsum(e)


def _estimate_curvature(prev_xytheta, curr_xytheta):
    """
    Estimate curvature at curr point from previous measured point.
    prev_xytheta and curr_xytheta are (x, y, theta) tuples.
    theta in radians.
    """
    x_prev, y_prev, th_prev = prev_xytheta
    x_curr, y_curr, th_curr = curr_xytheta
    dx = x_curr - x_prev
    dy = y_curr - y_prev
    ds = np.hypot(dx, dy)
    if ds < 1e-6:
        return 0.0
    # angle difference normalized
    dth = (th_curr - th_prev + np.pi) % (2 * np.pi) - np.pi
    return dth / ds


def _clothoid_path_from_states(start_state, target_state, k0=0.0, k1=0.0):
    """
    Use pyclothoids.SolveG2 to produce clothoid pieces connecting (x0,y0,theta0,k0)
    to (x1,y1,theta1,k1).

    start_state: [x0, y0, v0, theta0]
    target_state: [x1, y1, v1, theta1]

    Returns:
      pieces: list of 3 clothoid objects
      S_total: total arc-length
      lengths: list of segment lengths
    """
    x0, y0, v0, theta0 = start_state
    x1, y1, v1, theta1 = target_state

    # Solve G2: returns three clothoid segments (c1, c2, c3).
    pieces = SolveG2(x0, y0, theta0, k0, x1, y1, theta1, k1)
    pieces = list(pieces)
    # Some versions return a tuple/list already; ensure indexing works
    if not hasattr(pieces, "__iter__"):
        raise RuntimeError("SolveG2 returned unexpected type. Adjust code for your pyclothoids version.")

    lengths = [seg.length for seg in pieces]
    S_total = float(np.sum(lengths))
    return pieces, S_total, lengths


def _eval_path_xytheta(pieces, lengths, s_vals):
    """
    Evaluate the clothoid path at a list/array of arc-lengths s_vals in [0, S_total].
    Returns Nx2 array of (x,y).
    """
    out = np.zeros((len(s_vals), 4))
    seg_cum = np.concatenate(([0.0], np.cumsum(lengths)))
    for i, s in enumerate(s_vals):
        # clamp
        if s <= 0:
            seg_index = 0
            local_s = 0.0
        elif s >= seg_cum[-1]:
            seg_index = len(lengths) - 1
            local_s = lengths[-1]
        else:
            # find segment index such that seg_cum[idx] <= s < seg_cum[idx+1]
            seg_index = int(np.searchsorted(seg_cum, s, side='right') - 1)
            local_s = s - seg_cum[seg_index]
        seg = pieces[seg_index]
        out[i, 0] = float(seg.X(local_s))
        out[i, 1] = float(seg.Y(local_s))
        out[i, 2] = float(seg.Theta(local_s))
        out[i, 3] = float(seg.ThetaD(local_s))
    return out


def _apply_endpoint_correction(x, y, v, a, j, theta, omega, start_state, target_state, times):
    """
    Smoothly corrects the trajectory so that endpoints match exactly.
    Does linear interpolation of offsets.
    """
    N = len(x)
    alpha = np.linspace(0.0, 1.0, N)

    x0, y0, v0, theta0 = start_state
    x1, y1, v1, theta1 = target_state

    # offsets at both ends
    dx0 = x0 - x[0]
    dy0 = y0 - y[0]
    dv0 = v0 - v[0]

    dx1 = x1 - x[-1]
    dy1 = y1 - y[-1]
    dv1 = v1 - v[-1]
    dtheta0 = theta0 - theta[0]
    dtheta1 = theta1 - theta[-1]

    # correct
    x_corr = x + (1 - alpha) * dx0 + alpha * dx1
    y_corr = y + (1 - alpha) * dy0 + alpha * dy1
    v_corr = v + (1 - alpha) * dv0 + alpha * dv1
    theta_corr = theta + (1 - alpha) * dtheta0 + alpha * dtheta1

    # explicitly fix endpoints
    x_corr[0], y_corr[0], v_corr[0], theta_corr[0] = x0, y0, v0, theta0
    x_corr[-1], y_corr[-1], v_corr[-1], theta_corr[-1] = x1, y1, v1, theta1

    return x_corr, y_corr, v_corr, theta_corr


def reconstruct_gap(start_state, target_state, last_available_input, next_available_input, missing_times,
                    prev_state=None, post_state=None, k0=0.0, k1=0.0,
                    degree=6, lambda_jerk=1.0, lambda_vel=100.0, lambda_acc=10.0,
                    lambda_beta=1.0, beta_init=None, verbose=False):
    """
    Reconstruct a gap trajectory using:
      - G2 clothoid (pyclothoids.SolveG2) for geometry
      - monotone Bernstein-based s(t) parameterization (monotone by cumulative-exp reparam)
      - objective minimizing integrated jerk^2 with boundary penalties

    Handles:
      start_state, target_state = [x, y, speed, angle]
      last_available_input, next_available_input = [accel, angle_vel]  # accel is longitudinal accel
      missing_times: list or 1D-array of timestamps (must be in increasing order)
              
    Minimal implementation EXACTLY for the timeline:
      missing_times[0]  -> last_input AND start_state (measured)
      missing_times[1]  -> first interior gap sample (no measurement)
      ...
      missing_times[-2] -> last interior gap sample (no measurement)
      missing_times[-1] -> next_input AND target_state (measured)

    Keyword args:
      degree: Bernstein polynomial degree (common: 5..8). Higher = more DOF.
      lambda_jerk: weight for jerk^2 integral
      lambda_vel: weight for endpoint speed matching
      lambda_acc: weight for endpoint acceleration matching (uses last_available_input / next_available_input)
      beta_init: initial guess for beta (size degree+1). If None, zeros.
      verbose: print some solver output

    Returns:
      dict with keys: times, s, x, y, v, a, jerk, alpha, beta, success, message
    """
    times = np.asarray(missing_times).astype(float)
    if times.ndim != 1:
        raise ValueError("missing_times must be 1D array-like.")
    if len(times) < 2:
        raise ValueError("Need at least 2 times to reconstruct.")
        
    # indices by convention
    idx_start = 0
    idx_target = len(times) - 1

    # Determine prev/post curvature estimates if possible
    if prev_state is not None:
        k0 = _estimate_curvature((prev_state[0], prev_state[1], prev_state[3]), 
                                 (start_state[0], start_state[1], start_state[3]))
    if post_state is not None:
        k1 = _estimate_curvature((target_state[0], target_state[1], target_state[3]), 
                                 (post_state[0], post_state[1], post_state[3]))
    
    # 1) Build clothoid path
    try:
        pieces, S_total, lengths = _clothoid_path_from_states(start_state, target_state, k0=k0, k1=k1)
    except Exception as e:
        raise RuntimeError("Error constructing clothoid with SolveG2: " + str(e))

    # 2) Build Bernstein basis on normalized time domain [0,1]
    # define s(t) over [t_start, t_target] where t_start = times[0], t_target = times[-1]
    t_start = times[idx_start]
    t_target = times[idx_target]
    t_norm = (times - t_start) / (t_target - t_start)  # in [0,1] across the full array

    n_coeff = degree + 1
    # beta0 = np.zeros(n_coeff) if beta_init is None else np.asarray(beta_init, dtype=float)
    equal_incr = 1.0 / n_coeff
    beta0 = np.log(np.ones(n_coeff) * equal_incr) # warm start

    # 3) objective helpers: finite-difference derivatives with robust np.gradient
    def compute_from_beta(beta):
        alpha = _monotone_alpha_from_beta(beta)
        # B = bernstein_basis(degree, t_norm)
        # B1 = B[-1, :]
        # s_norm = B.dot(alpha)
        # denom = float(np.dot(B1, alpha))
        # if denom <= 0:
        #     denom = 1e-12
        # s = S_total * (s_norm / denom)
        # v = np.gradient(s, times)
        # a = np.gradient(v, times)
        # j = np.gradient(a, times)
        # return alpha, s, v, a, j
        
        # Bernstein and derivatives
        B, B1, B2, B3 = bernstein_derivatives(degree, t_norm)
        # --- normalization that ensures s(0)=0 and s(1)=S_total ---
        B0_dot_alpha = float(np.dot(B[0, :], alpha))   # B(0)·alpha
        B1_dot_alpha = float(np.dot(B[-1, :], alpha))  # B(1)·alpha
        denom = B1_dot_alpha - B0_dot_alpha
        if abs(denom) < 1e-12:
            denom = 1e-12
        # numerator: B @ alpha - B(0)·alpha   (shape (N,))
        s_norm = B.dot(alpha) - B0_dot_alpha
        
        # analytical s, v, a, j
        s = S_total * s_norm  / denom
        v = S_total * np.dot(B1, alpha) / denom
        a = S_total * np.dot(B2, alpha) / denom
        j = S_total * np.dot(B3, alpha) / denom
        return alpha, s, v, a, j

    # Extract boundary targets
    v0_target = float(start_state[2])
    v1_target = float(target_state[2])
    a0_target = float(last_available_input[0])   # accel before gap
    a1_target = float(next_available_input[0])   # accel after gap

    # objective: discretized integrated jerk^2 + boundary penalties
    T_ref = 1.0
    def _beta_smoothness_penalty(beta):
        # second-difference penalty: sum_i (beta[i+1] - 2*beta[i] + beta[i-1])^2
        if beta.size < 3:
            return 0.0
        d2 = beta[2:] - 2.0 * beta[1:-1] + beta[:-2]
        return float(np.sum(d2 * d2))

    def objective(beta):
        alpha, s, v, a, j = compute_from_beta(beta)
        # approximate integral of j^2 over time with trapezoidal rule
        j2 = j ** 2
        jerk_pen = np.trapz(j2, times)
        acc_effort_pen = np.trapz(a ** 2, times)
        # vel_pen = (v[0] - v0_target) ** 2 + (v[-1] - v1_target) ** 2
        # acc_pen = (a[0] - a0_target) ** 2 + (a[-1] - a1_target) ** 2
            
        # beta smoothness regularizer (stronger for longer gaps)
        T_gap = float(times[-1] - times[0])
        scale = max(1.0, np.ceil(T_gap / T_ref))   # e.g., if T_ref=1s then gap 2s => scale 2.0
        # beta_pen = _beta_smoothness_penalty(beta)
        beta_pen = np.sum(beta ** 2)
        
        # obj = lambda_jerk * jerk_pen + lambda_vel * vel_pen + \
        #         lambda_acc * acc_pen + lambda_beta * scale * beta_pen + \
        #         0.1 * lambda_acc * acc_effort_pen
        
        obj = lambda_jerk * jerk_pen + lambda_beta * scale * beta_pen + lambda_acc * acc_effort_pen
        return obj

    # optimization
    if verbose:
        print("Optimizing monotone s(t) on [{:.6f}, {:.6f}], N={}, S_total={:.4f}, k0={:.4f}, k1={:.4f}".
              format(t_start, t_target, len(times), S_total, k0, k1))
    
    # res = minimize(objective, beta0, method="L-BFGS-B", options={'maxiter': 200, 'disp': verbose})
    # beta_opt = res.x
    
    beta_opt, al_res = augmented_lagrangian(
        objective_fn=objective,
        beta0=beta0,
        compute_from_beta_fn=compute_from_beta,
        clothoid_pieces=pieces,
        clothoid_lengths=lengths,
        v0_target=v0_target,
        v1_target=v1_target,
        a0_target=a0_target,
        a1_target=a1_target,
        times=times,
        max_outer=20,
        mu0=100.0,
        mu_factor=5.0,
        tol_constraint=1e-04,
        verbose=verbose
    )
    
    alpha_opt, s_opt, v_opt, a_opt, j_opt = compute_from_beta(beta_opt)
    

    # Map s -> (x,y)
    xytheta = _eval_path_xytheta(pieces, lengths, s_opt)
    x = xytheta[:, 0]
    y = xytheta[:, 1]
    theta = xytheta[:, 2]
    kappa = xytheta[:, 3]
    omega = kappa * v_opt
    
    # # OPTIONAL: endpoint correction
    # _, _, v_opt2, _ = _apply_endpoint_correction(
    #     x, y, v_opt, a_opt, j_opt, theta, omega,
    #     start_state, target_state, times
    # )
    # a_opt2 = np.gradient(v_opt2, times)
    
    
    
    import matplotlib.pyplot as plt
    
    plt.figure('xy')
    plt.plot(x, y)
    plt.scatter(start_state[0], start_state[1], color='black')
    plt.scatter(target_state[0], target_state[1], color='red')
    
    plt.figure('tv')
    plt.plot(times, v_opt)
    plt.scatter(times[0], start_state[-2], color='black')
    plt.scatter(times[-1], target_state[-2], color='red')
    
    plt.figure('ta')
    plt.plot(times, a_opt)
    plt.scatter(times[0], last_available_input[0], color='black')
    plt.scatter(times[-1], next_available_input[0], color='red')
    
    plt.figure('t-theta')
    plt.plot(times, theta)
    plt.scatter(times[0], start_state[-1], color='black')
    plt.scatter(times[-1], target_state[-1], color='red')
    
    plt.figure('t-omega')
    plt.plot(times, omega)
    plt.scatter(times[0], last_available_input[1], color='black')
    plt.scatter(times[-1], next_available_input[1], color='red')
    
    sys.exit(1)
    
    

    out = {
        'times': times,
        's': s_opt,
        'x': x,
        'y': y,
        'v': v_opt,
        'a': a_opt,
        'jerk': j_opt,
        'theta': theta,
        'omega': omega, 
        'alpha': alpha_opt,
        'beta': beta_opt,
        # 'success': bool(res.success),
        # 'message': res.message,
        # 'optimization_result': res,
        'S_total': S_total,
        'clothoid_lengths': lengths,
    }
    return out


def estimate_curvature(df, idx, window=3):
    """
    Estimate curvature k at frame idx from xy trajectory using finite-difference.
    If idx is near edge of valid region (e.g., missing data), use available samples.
    """
    # Extract local window
    lo = max(idx - window, 0)
    hi = min(idx + window + 1, len(df))
    pts = df.loc[lo:hi, ['x', 'y']].values

    if len(pts) < 3:
        return 0.0

    # Local polyfit-based curvature estimate
    # Fit parametric curve s -> x(s) and y(s)
    s = np.linspace(0, 1, len(pts))
    px = np.polyfit(s, pts[:,0], 2)
    py = np.polyfit(s, pts[:,1], 2)

    dx  = np.polyval(np.polyder(px,1), s)
    dy  = np.polyval(np.polyder(py,1), s)
    ddx = np.polyval(np.polyder(px,2), s)
    ddy = np.polyval(np.polyder(py,2), s)

    # Curvature κ = (dx*ddy - dy*ddx)/(dx²+dy²)^(3/2)
    k = (dx*ddy - dy*ddx) / (dx*dx + dy*dy)**1.5
    return float(np.median(k))


def augmented_lagrangian(objective_fn, beta0, compute_from_beta_fn, times,
                         v0_target, v1_target, a0_target, a1_target,
                         clothoid_pieces, clothoid_lengths,
                         a_min=-3.0, a_max=3.0,
                         omega_min=-0.5, omega_max=0.5,
                         max_outer=12,
                         mu0=50.0,
                         mu_factor=10.0,
                         tol_constraint=1e-6,
                         verbose=False):
    """
    Augmented Lagrangian with:
      - equality constraints: v(0), v(T), a(0), a(T)
      - inequality constraints: a_min <= a(t) <= a_max
    """

    beta = np.array(beta0, dtype=float)

    # Lagrange multipliers
    lam_v = np.zeros(2); lam_a = np.zeros(2)
    mu_v = 1e3; mu_a = 50.0
    lam_ineq = None                     # allocated after first eval

    mu = float(mu0)
    
    v_scale = max(1.0, (abs(v0_target)+abs(v1_target))/2.0)
    a_scale = max(1.0, (abs(a0_target)+abs(a1_target))/2.0)

    def eq_residuals(beta):
        _, _, v, a, _ = compute_from_beta_fn(beta)
        c_v = np.array([(v[0] - v0_target)/v_scale, (v[-1] - v1_target)/v_scale])
        c_a = np.array([(a[0] - a0_target)/a_scale, (a[-1] - a1_target)/a_scale])
        return c_v, c_a

    def ineq_residuals(beta):
        nonlocal lam_ineq
        _, s, v, a, _ = compute_from_beta_fn(beta)
        
        # geometric evaluation
        xytheta = _eval_path_xytheta(clothoid_pieces, clothoid_lengths, s)
        kappa = xytheta[:, 3]
        omega = kappa * v

        # bounds
        g_a_upper = a - a_max
        g_a_lower = a_min - a
    
        g_w_upper = omega - omega_max
        g_w_lower = omega_min - omega
    
        # combined inequality vector
        g = np.concatenate([
            g_a_upper, g_a_lower,
            g_w_upper, g_w_lower
        ])

        # initialize lambdas if first call
        if lam_ineq is None:
            lam_ineq = np.zeros_like(g)

        return g

    def augmented_objective(beta_vec):
        base = float(objective_fn(beta_vec))

        # constraints
        c_v, c_a = eq_residuals(beta)
        cineq = ineq_residuals(beta_vec)

        # only penalize positive violations for inequalities
        pos = np.maximum(0.0, cineq)

        aug = (
            lam_v.dot(c_v) + 0.5*mu_v*(c_v@c_v) + 
            lam_a.dot(c_a) + 0.5*mu_a*(c_a@c_a) +
            np.dot(lam_ineq, pos) +
            0.5 * mu * np.dot(pos, pos)
        )

        return base + aug

    for outer in range(max_outer):
        res = minimize(augmented_objective, beta, method='L-BFGS-B',
                       options={'maxiter': 200, 'disp': False})

        beta = res.x

        # compute constraints
        c_v, c_a = eq_residuals(beta)
        cineq = ineq_residuals(beta)

        pos = np.maximum(0.0, cineq)

        max_eq = np.max(np.abs(c_v))
        max_ineq = np.max(pos)

        if verbose:
            print(f"[AL] iter {outer:02d}  mu={mu:.2e}  "
                  f"max_eq={max_eq:.3e}  max_ineq={max_ineq:.3e}")

        if max(max_eq, max_ineq) < tol_constraint:
            if verbose:
                print("[AL] constraints satisfied.")
            break

        # multiplier updates
        lam_v = lam_v + mu_v * c_v
        lam_a = lam_a + mu_a * c_a
        lam_ineq = lam_ineq + mu * pos

        # tighten penalty
        mu_v *= mu_factor
        mu_a *= mu_factor
        mu *= mu_factor

    return beta, res