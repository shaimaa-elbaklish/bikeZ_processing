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
import cvxpy as cp

from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from pyclothoids import Clothoid, SolveG2

from _constants import SKIP_KALMAN_FILTERING_MAX_GAP
from tools_filtering import calculate_features, boundAnglePositive

# #############################################################################
# METHODS
# #############################################################################
def calculate_kalman_filtered_trajectory(veh_df: pd.DataFrame, Q_t: np.ndarray, R_t: np.ndarray, 
                                         first_frame: int, last_frame: int, fps: float = 25.0):
    feat_veh_df = calculate_features(veh_df, fps)      
    feat_veh_df = _reconstruct_gaps(feat_veh_df, first_frame, last_frame, fps)
    
    
    
    C_t = np.diag([1, 1, 1, 1])
    I = np.eye(4)
    
    
    sys.exit(1)
    

def _reconstruct_gaps(veh_df: pd.DataFrame, first_frame: int, last_frame: int, fps: float = 25.0):
    veh_df = veh_df.reset_index().drop(columns='index')
    veh_df['frame_nr_diff'] = veh_df['frame_nr'].diff()
    gaps_df = veh_df[veh_df['frame_nr_diff'] > 1]
    for idx, _ in gaps_df.iterrows():
        before_gap = veh_df.loc[idx-1]
        after_gap = veh_df.loc[idx]
        gap_len = after_gap['frame_nr'] - before_gap['frame_nr'] - 1
        if gap_len < SKIP_KALMAN_FILTERING_MAX_GAP:
            continue
        N_total = gap_len + 2        # includes start & end
        # Extract boundary states
        x0, y0 = before_gap['x'], before_gap['y']
        th0, av0 = before_gap['angle_estimation'], before_gap['angle_vel_estimation']
        v0, a0 = before_gap['speed'], before_gap['a']
        x1, y1 = after_gap['x'], after_gap['y']
        th1, av1 = after_gap['angle_estimation'], after_gap['angle_vel_estimation']
        v1, a1 = after_gap['speed'], after_gap['a']
        # Estimate jerk at boundaries
        a_prev = veh_df.loc[idx-2]['a']
        a_next = veh_df.loc[idx+1]['a']
        j0 = (a0 - a_prev)*fps
        j1  = (a_next - a1)*fps
        # Estimate curvature at boundaries
        k0 = _estimate_curvature(veh_df, idx-1)
        k1 = _estimate_curvature(veh_df, idx+1)
        # Build the G2 clothoid path
        segments = _construct_clothoid_G2(x0, y0, th0, k0, x1, y1, th1, k1, fallback=True)
        # Speed interpolation (smooth quartic + constraints)
        print(j0, j1)
        # speed, accel = _interpolate_speed_constrained(v0, v1, a0, a1, j0, j1, av0, av1, kappa, N_total, 1.0/fps)
        # coeffs = heptic_c3_minjerk_qp(v0, v1, a0, a1, j0, j1, T=(N_total-1)/fps, N=N_total)
        L_total = 0
        for seg in segments:
            L_total += seg.length
        coeffs = solve_stime_qp(
            s0=0, s1=L_total,
            v0=v0, v1=v1, a0=a0, a1=a1, j0=j0, j1=j1,
            T=(N_total-1)/fps, K=N_total, degree=8, solver=cp.MOSEK
        )
        missing_frames = np.arange(before_gap['frame_nr'], after_gap['frame_nr']+1, 1)
        missing_times = missing_frames / fps
        # speed, accel, jerk = eval_poly(coeffs, missing_times - missing_times[0])
        s_vals, speed, accel, jerk = eval_poly_s(coeffs, missing_times - missing_times[0])
        x, y, th, kappa = eval_clothoid_segments(segments, s_vals)
        # Dynamics
        angle_vel = speed * kappa
        
        print(missing_frames.shape)
        print(x.shape, y.shape)
        print(th.shape)
        print(speed.shape)
        print(accel.shape)
        print(th.shape)
        print(angle_vel.shape)
        
        import matplotlib.pyplot as plt
        
        plt.figure('xy')
        plt.scatter(veh_df['x'], veh_df['y'], s=1)
        plt.plot(x, y, color='red')
        
        plt.figure('speed')
        plt.scatter(veh_df['frame_nr'], veh_df['speed'], s=1)
        plt.plot(missing_frames, speed, color='red')
        
        plt.figure('accel')
        plt.scatter(veh_df['frame_nr'], veh_df['a'], s=1)
        plt.plot(missing_frames, accel, color='red')
        
        plt.figure('jerk')
        plt.scatter(veh_df['frame_nr'], veh_df['a'].diff()*fps, s=1)
        # plt.plot(missing_frames[:-1], np.diff(accel)*fps, color='red')
        plt.plot(missing_frames, jerk, color='red')
        
        plt.figure('angle')
        plt.scatter(veh_df['frame_nr'], veh_df['angle_estimation'], s=1)
        plt.plot(missing_frames, th, color='red')
        
        plt.figure('angle_vel')
        plt.scatter(veh_df['frame_nr'], veh_df['angle_vel_estimation'], s=1)
        plt.plot(missing_frames, angle_vel, color='red')
        
        
        sys.exit(1)
        
        veh_missing_df = pd.DataFrame(np.column_stack([missing_times, missing_frames, x[1:], y[1:], speed[1:], th[1:], accel[1:], angle_vel[1:]]),
                                      columns=['time', 'frame_nr', 'x', 'y', 'speed', 'angle_estimation', 'a', 'angle_vel_estimation'])
        print(veh_missing_df.head())
        sys.exit(1)


def _f_dyn(x, u, dt):
    # x_arr = [x, y, v, theta]
    # u = [accel, omega]
    return np.array([
        x[0] + dt*x[2]*np.cos(x[3]),
        x[1] + dt*x[2]*np.sin(x[3]),
        max(0, x[2] + dt*u[0]),
        boundAnglePositive(x[3] + dt*u[1], "rad")
    ])


def _A_jacobian(x, u, dt):
    return np.array([
        [1, 0, dt*np.cos(x[-1]), -dt*x[2]*np.sin(x[-1])],
        [0, 1, dt*np.sin(x[-1]), dt*x[2]*np.cos(x[-1])],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


def _B_jacobian(x, u, dt):
    return np.array([[0], [0], [dt], [dt]])


def _estimate_curvature(df, idx, window=3):
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


def _construct_clothoid_G2(x0, y0, th0, k0, x1, y1, th1, k1, fallback=True):
    """
    Try constructing a G2 clothoid between two states.
    Returns a list of Clothoid segments (3 segments for G2).
    """
    try:
        segments = SolveG2(x0, y0, th0, k0,
                           x1, y1, th1, k1)
        # segments is a tuple/list of 3 Clothoids
        return list(segments)
    except Exception as e:
        if not fallback:
            raise e
        # fallback: G1
        print('Fallback!!')
        C = Clothoid.G1Hermite(x0, y0, th0,
                               x1, y1, th1)
        return [C]
    
    
def _sample_clothoid_segments(segments, n_samples):
    """
    Sample a sequence of G2 clothoid segments uniformly in arc length.
    """
    lengths = [seg.length for seg in segments]
    L = sum(lengths)

    s_vals = np.linspace(0, L, n_samples)

    x = np.zeros(n_samples)
    y = np.zeros(n_samples)
    th = np.zeros(n_samples)
    kappa = np.zeros(n_samples)

    # Walk through segments
    cum = np.cumsum([0] + lengths)
    for i, s in enumerate(s_vals):
        # find the segment index
        seg_id = np.searchsorted(cum, s, side='right') - 1
        seg_id = min(seg_id, len(segments)-1)

        seg = segments[seg_id]
        s_local = s - cum[seg_id]

        x[i] = seg.X(s_local)
        y[i] = seg.Y(s_local)
        th[i] = seg.Theta(s_local)
        kappa[i] = seg.ThetaD(s_local)

    return x, y, th, kappa


def heptic_c3_minjerk_qp(v0, v1, a0, a1, j0, j1, T, N, a_lb=-3.0, a_ub=3.0, lambda_slack=100.0):
    # Coefficients c0..c6
    c = cp.Variable(7)
    s = cp.Variable(N, nonneg=True)

    # Discretize t for integral
    t = np.linspace(0, T, N)
    dt = t[1]-t[0]
    
    # v'(t) = c1 + 2c2 t + 3c3 t^2 + 4c4 t^3 + 5c5 t^4 + 6c6 t^5
    # v''(t) = 2 c2 + 6 c3 t + 12 c4 t^2 + 20 c5 t^3 + 30 c6 t^4
    vp = c[1] + 2*c[2]*t + 3*c[3]*t**2 + 4*c[4]*t**3 + 5*c[5]*t**4 + 6*c[6]*t**5
    vpp = 2*c[2] + 6*c[3]*t + 12*c[4]*t**2 + 20*c[5]*t**3 + 30*c[6]*t**4

    # Objective: sum of squared v'''(t) * dt ~ integral
    objective = cp.Minimize(cp.sum_squares(vpp)*dt + lambda_slack*cp.sum_squares(s)) # 25*cp.sum_squares(vp)*dt

    # Linear equality constraints
    constraints = [
        c[0] == v0,
        c[0] + c[1]*T + c[2]*T**2 + c[3]*T**3 + c[4]*T**4 + c[5]*T**5 + c[6]*T**6 == v1,
        c[1] == a0,
        c[1] + 2*c[2]*T + 3*c[3]*T**2 + 4*c[4]*T**3 + 5*c[5]*T**4 + 6*c[6]*T**5 == a1,
        2*c[2] == j0,
        2*c[2] + 6*c[3]*T + 12*c[4]*T**2 + 20*c[5]*T**3 + 30*c[6]*T**4 == j1
    ]
    # Soft acceleration bounds:
    constraints += [
        vp >= a_lb - s,
        vp <= a_ub + s
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK)

    return c.value


# Evaluate polynomial at times:
def eval_poly(coeffs, t):
    coeffs_a = np.array([i * c for i, c in enumerate(coeffs)][1:])  # c1 + 2*c2 t + ...
    coeffs_j = np.array([i * c for i, c in enumerate(coeffs_a)][1:])  # 2*c2 + 6*c3 t + ...
    
    speed = np.polyval(coeffs[::-1], t)  # because np.polyval expects highest-first
    accel = np.polyval(coeffs_a[::-1], t)
    jerk = np.polyval(coeffs_j[::-1], t)
    return speed, accel, jerk


def _interpolate_speed_constrained(v0, v1, a0, a1, j0, j1, av0, av1, kappa,
                                   N, dt, a_lb=-3.0, a_ub=3.0, av_lb=-0.5, av_ub=0.5):
    """
    Smooth speed interpolation subject to accel bounds and endpoint constraints.
    
    v(t) spline is represented by N internal control accelerations a[i].
    Integrates acceleration → speed using trapezoidal rule.
    Minimizes jerk magnitude between successive accelerations.
    """

    L = N                    # number of accel samples
    M = N + 1                # number of speed samples

    # -------------------------------
    # Integration
    # -------------------------------
    def accel_to_speed(a):
        """Forward Euler integration into speed."""
        v = np.zeros(M)
        v[0] = v0
        v[1:] = v0 + np.cumsum(dt * a)
        return v

    # -------------------------------
    # Cost (jerk squared)
    # -------------------------------
    def jerk_cost(a):
        j = np.diff(a) / dt
        return np.sum(j * j)

    # -------------------------------
    # Constraints
    # -------------------------------
    def final_speed_constraint(a):
        v = accel_to_speed(a)
        return v[N-1] - v1

    def init_acc_constraint(a):
        return a[0] - a0

    def final_acc_constraint(a):
        return a[N-1] - a1
    
    def init_jerk_constraint(a):
        # (a[1] - a[0]) / dt = j0  →  a[1] - a[0] - j0*dt = 0
        return (a[1] - a[0]) - j0*dt
    
    def final_jerk_constraint(a):
        # (a[N-1] - a[N-2]) / dt = jT
        return (a[N-1] - a[N-2]) - j1*dt
    
    def init_angle_vel_constraint(a):
        return accel_to_speed(a)[0] * kappa[0] - av0
    
    def final_angle_vel_constraint(a):
        return accel_to_speed(a)[N-1] * kappa[N-1] - av1

    constraints = [
        {'type': 'eq', 'fun': final_speed_constraint},
        {'type': 'eq', 'fun': init_acc_constraint},
        {'type': 'eq', 'fun': final_acc_constraint},
        {'type': 'eq', 'fun': init_jerk_constraint},
        {'type': 'eq', 'fun': final_jerk_constraint},
        # {'type': 'eq', 'fun': init_angle_vel_constraint},
        # {'type': 'eq', 'fun': final_angle_vel_constraint},    
    ]
    # for i in range(N):
    #     constraints.append({
    #         'type': 'ineq',
    #         'fun': lambda a, i=i: accel_to_speed(a)[i] * kappa[i] - av_lb
    #     })
    #     constraints.append({
    #         'type': 'ineq',
    #         'fun': lambda a, i=i: av_ub - accel_to_speed(a)[i] * kappa[i]
    #     })

    bounds = [(a_lb, a_ub)] * L

    # initial guess
    a_guess = np.linspace(a0, a1, L)

    res = minimize(
        jerk_cost,
        a_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 250, 'ftol': 1e-9}
    )

    if not res.success:
        print("⚠ SLSQP failed — using linear accel ramp.")
        a_opt = a_guess
    else:
        a_opt = res.x

    v_full = accel_to_speed(a_opt)

    # usable speed, same length as geometry
    v_use = v_full[:-1]

    return v_use, a_opt


def build_Q_j2(d, T):
    """Build Q for J = ∫ (s'''(t))^2 dt for polynomial degree d."""
    Q = np.zeros((d+1, d+1), dtype=float)
    for i in range(3, d+1):
        for j in range(3, d+1):
            power = (i-3) + (j-3)
            Q[i, j] = (i*(i-1)*(i-2)) * (j*(j-1)*(j-2)) * (T**(power+1)) / (power+1)
    return Q

def build_endpoint_constraints(d, T, s0, s1, v0, v1, a0, a1, j0, j1):
    """
    Return (A, b) enforcing s(0)=s0, s(T)=s1, v(0)=v0, v(T)=v1,
    a(0)=a0, a(T)=a1, j(0)=j0, j(T)=j1.
    A shape: (8, d+1)
    """
    A = np.zeros((8, d+1), dtype=float)
    b = np.zeros(8, dtype=float)

    # s(0) = c0 = s0
    A[0,0] = 1.0; b[0] = s0

    # s(T)
    A[1,:] = np.array([T**i for i in range(d+1)])
    b[1] = s1

    # v(0) = c1 = v0
    if d >= 1:
        A[2,1] = 1.0
    b[2] = v0

    # v(T) = sum i*c_i T^{i-1}
    deriv_T = np.array([0.0] + [i * T**(i-1) for i in range(1, d+1)])
    A[3,:] = deriv_T
    b[3] = v1

    # a(0) = 2*c2 = a0
    if d >= 2:
        A[4,2] = 2.0
    b[4] = a0

    # a(T)
    acoef = np.zeros(d+1)
    for i in range(2, d+1):
        acoef[i] = i*(i-1) * (T**(i-2))
    A[5,:] = acoef
    b[5] = a1

    # j(0) = 6*c3 = j0
    if d >= 3:
        A[6,3] = 6.0
    b[6] = j0

    # j(T)
    jcoef = np.zeros(d+1)
    for i in range(3, d+1):
        jcoef[i] = i*(i-1)*(i-2) * (T**(i-3))
    A[7,:] = jcoef
    b[7] = j1

    return A, b

def solve_stime_qp(
        s0, s1,                         # s at t=0 and t=T (you can set s1 = s0 + L_total)
        v0, v1, a0, a1, j0, j1,         # endpoint kinematics
        T, K,                           # duration and #points (use T=(N_total-1)/fps, K=N_total)
        degree=8,                       # polynomial degree (recommend 8 -> 9 coeffs)
        solver=cp.OSQP,
        verbose=False
    ):
    """
    Solve convex QP for coefficients c0..c_degree of s(t) minimizing ∫ (s'''(t))^2 dt
    subject to hard endpoint equalities for s,v,a,j.
    Returns coeffs numpy array length degree+1.
    NOTE: If degree == 7 and you give 8 hard constraints, the solution is unique
          and minimization is irrelevant. Use degree >= 8 to retain freedom.
    """
    d = degree
    Q = build_Q_j2(d, T)
    # variable
    c = cp.Variable(d+1)

    # objective: 0.5*c^T Q c  (cvxpy prefers cp.quad_form)
    # scale Q small to help numerics? not needed usually
    obj = 0.5 * cp.quad_form(c, Q)

    # constraints
    A, b = build_endpoint_constraints(d, T, s0, s1, v0, v1, a0, a1, j0, j1)
    constraints = [A[i,:] @ c == b[i] for i in range(A.shape[0])]
    
    # positive speed constraints
    ts = np.linspace(0, T, K)  # INCLUDE interior points
    eps = 0              # minimum allowable speed
    
    for t in ts:
        # v(t) = sum_{i=1..d} i * c[i] * t^(i-1)
        s_t = sum(c[i] * (t**i) for i in range(0, d + 1))
        constraints.append(s_t <= s1)
    
    # for t in ts:
    #     # v(t) = sum_{i=1..d} i * c[i] * t^(i-1)
    #     v_t = sum(i * c[i] * (t ** (i - 1)) for i in range(1, d + 1))
    #     constraints.append(v_t >= eps)
    
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=solver, verbose=verbose)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError("QP failed: " + str(prob.status))

    return np.array(c.value).flatten()


def eval_poly_s(coeffs, t):
    coeffs_v = np.array([i * c for i, c in enumerate(coeffs)][1:]) # c1 + 2*c2 t + ...
    coeffs_a = np.array([i * c for i, c in enumerate(coeffs_v)][1:]) # 2*c2 + 6*c3 t + ...
    coeffs_j = np.array([i * c for i, c in enumerate(coeffs_a)][1:])  
    
    arc_len = np.polyval(coeffs[::-1], t)
    speed = np.polyval(coeffs_v[::-1], t)  # because np.polyval expects highest-first
    accel = np.polyval(coeffs_a[::-1], t)
    jerk = np.polyval(coeffs_j[::-1], t)
    return arc_len, speed, accel, jerk


def eval_clothoid_segments(segments, s_vals):
    """
    Sample a sequence of G2 clothoid segments uniformly in arc length.
    """
    lengths = [seg.length for seg in segments]
    L = sum(lengths)
    
    s_vals = np.round(s_vals, decimals=6)
    
    print(np.amin(s_vals), np.amax(s_vals))
    print(s_vals[0], s_vals[1])
    n_samples = len(s_vals)

    x = np.zeros(n_samples)
    y = np.zeros(n_samples)
    th = np.zeros(n_samples)
    kappa = np.zeros(n_samples)

    # Walk through segments
    cum = np.cumsum([0] + lengths)
    for i, s in enumerate(s_vals):
        # find the segment index
        seg_id = np.searchsorted(cum, s, side='right') - 1
        seg_id = min(seg_id, len(segments)-1)

        seg = segments[seg_id]
        s_local = s - cum[seg_id]

        x[i] = seg.X(s_local)
        y[i] = seg.Y(s_local)
        th[i] = seg.Theta(s_local)
        kappa[i] = seg.ThetaD(s_local)

    return x, y, th, kappa


