import numpy as np
from numba import njit

from .utils import steering_constraint, accl_constraints

@njit(cache=True)
def vehicle_dynamics_st(x: np.ndarray, u_init: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Single Track Vehicle Dynamics with linearly saturated lateral tire forces.
    From https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/blob/master/vehicleModels_commonRoad.pdf, section 7

        Args:
            x (numpy.ndarray (7, )): vehicle state vector (x0, x1, x2, x3, x4, x5, x6)
                x0: x position in global coordinates
                x1: y position in global coordinates
                x2: steering angle of front wheels
                x3: velocity in x direction
                x4: yaw angle
                x5: yaw rate
                x6: slip angle at vehicle center
            u (numpy.ndarray (2, )): control input vector (u1, u2)
                u1: steering angle velocity of front wheels
                u2: longitudinal acceleration
            params (np.ndarray (89, )): flat float32 parameter array in VehicleParameters declaration order; this model reads indices 0-17 and 88 (a dict raises a numba typing error):
                mu_y_peak (float): peak lateral force / normal load (index 88)
                mu (float): linear-response friction coefficient
                C_Sf (float): cornering stiffness of front wheels
                C_Sr (float): cornering stiffness of rear wheels
                lf (float): distance from center of gravity to front axle
                lr (float): distance from center of gravity to rear axle
                h (float): height of center of gravity
                m (float): mass of vehicle
                I (float): moment of inertia of vehicle, about Z axis
                s_min (float): minimum steering angle
                s_max (float): maximum steering angle
                sv_min (float): minimum steering velocity
                sv_max (float): maximum steering velocity
                v_switch (float): velocity above which the acceleration is no longer able to create wheel spin
                a_max (float): maximum allowed acceleration
                v_min (float): minimum allowed velocity
                v_max (float): maximum allowed velocity

        Returns:
            f (numpy.ndarray): right hand side of differential equations
    """
    # States
    X = x[0]
    Y = x[1]
    DELTA = x[2]
    V = x[3]
    PSI = x[4]
    PSI_DOT = x[5]
    BETA = x[6]
    # We have to wrap the slip angle to [-pi, pi]
    # BETA = np.arctan2(np.sin(BETA), np.cos(BETA))

    # gravity constant m/s^2
    g = 9.81
    mu = params[0]
    C_Sf = params[1]  # normalized front cornering stiffness [1/rad]
    C_Sr = params[2]  # normalized rear cornering stiffness [1/rad]
    lf = params[3]  # distance from spring mass center of gravity to front axle [m]  LENA
    lr = params[4]  # distance from spring mass center of gravity to rear axle [m]  LENB
    h = params[5]  # M_s center of gravity above ground [m]  HS
    m = params[6]  # vehicle mass [kg]  MASS
    I = params[7]  # moment of inertia for sprung mass in yaw [kg m^2]  IZZ
    s_min = params[8]  # minimum steering angle [rad]
    s_max = params[9]  # maximum steering angle [rad]
    sv_min = params[10]  # minimum steering velocity [rad/s]
    sv_max = params[11]  # maximum steering velocity [rad/s]
    v_switch = params[12]  # switching velocity [m/s]
    a_max = params[13]  # maximum absolute acceleration [m/s^2]
    v_min = params[14]  # minimum velocity [m/s]
    v_max = params[15]  # maximum velocity [m/s]
    # width = params[16]  # vehicle width [m]
    # length = params[17]  # vehicle length [m]

    # constraints
    u = np.array(
        [
            steering_constraint(
                DELTA,
                u_init[0],
                s_min,
                s_max,
                sv_min,
                sv_max,
            ),
            accl_constraints(
                V,
                u_init[1],
                v_switch,
                a_max,
                v_min,
                v_max,
            ),
        ]
    )
    # Controls
    STEER_VEL = u[0]
    ACCL = u[1]

    # switch to kinematic model for small velocities
    if V < 0.5:
        # wheelbase
        lwb = lf + lr
        BETA_HAT = np.arctan(np.tan(DELTA) * lr / lwb)
        BETA_DOT = (
            (1 / (1 + (np.tan(DELTA) * (lr / lwb)) ** 2))
            * (lr / (lwb * np.cos(DELTA) ** 2))
            * STEER_VEL
        )
        f = np.array(
            [
                V * np.cos(PSI + BETA_HAT),  # X_DOT
                V * np.sin(PSI + BETA_HAT),  # Y_DOT
                STEER_VEL,  # DELTA_DOT
                ACCL,  # V_DOT
                V * np.cos(BETA_HAT) * np.tan(DELTA) / lwb,  # PSI_DOT
                (1 / lwb)
                * (
                    ACCL * np.cos(BETA) * np.tan(DELTA)
                    - V * np.sin(BETA) * np.tan(DELTA) * BETA_DOT
                    + ((V * np.cos(BETA) * STEER_VEL) / (np.cos(DELTA) ** 2))
                ),  # PSI_DOT_DOT
                BETA_DOT,  # BETA_DOT
            ]
        )
    else:
        # Positive axle slip produces a positive lateral force. Keep the
        # existing small-angle slips and longitudinal load transfer; an
        # unloaded axle cannot generate tire force.
        wheelbase = lf + lr
        F_zf = max(0.0, m * (g * lr - ACCL * h) / wheelbase)
        F_zr = max(0.0, m * (g * lf + ACCL * h) / wheelbase)
        alpha_f = DELTA - BETA - lf * PSI_DOT / V
        alpha_r = -BETA + lr * PSI_DOT / V
        mu_y_peak = params[88]
        F_yf = min(
            max(mu * C_Sf * F_zf * alpha_f, -mu_y_peak * F_zf),
            mu_y_peak * F_zf,
        )
        F_yr = min(
            max(mu * C_Sr * F_zr * alpha_r, -mu_y_peak * F_zr),
            mu_y_peak * F_zr,
        )
        f = np.array(
            [
                V * np.cos(PSI + BETA),  # X_DOT
                V * np.sin(PSI + BETA),  # Y_DOT
                STEER_VEL,  # DELTA_DOT
                ACCL,  # V_DOT
                PSI_DOT,  # PSI_DOT
                (lf * F_yf - lr * F_yr) / I,  # PSI_DOT_DOT
                (F_yf + F_yr) / (m * V) - PSI_DOT,  # BETA_DOT
            ]
        )

    return f

@njit(cache=True)
def get_standardized_state_st(x: np.ndarray) -> np.ndarray:
    """[X, Y, steering_angle, speed, yaw, yaw_rate, beta] — index 6 is the SLIP ANGLE, not v_y."""
    return x
