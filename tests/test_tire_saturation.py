"""Physical limits and host/device parity for the saturated ST model."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from f1tenth_gym.envs.dynamic_models import F1TENTH_VEHICLE_PARAMETERS as BASE
from f1tenth_gym.envs.dynamic_models.single_track import vehicle_dynamics_st
from f1tenth_gym.envs.dynamic_models.jax import DynamicsParams, single_track


def derivatives(state, vehicle, acceleration=0.0):
    control = np.array([0.0, acceleration], dtype=np.float32)
    expected = vehicle_dynamics_st(state, control, vehicle.to_array())
    actual = jax.jit(single_track)(
        state, control, DynamicsParams.from_vehicle_parameters(vehicle)
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    return expected


def axle_forces(state, derivative, vehicle):
    """Recover axle forces from lateral momentum and yaw moment balance."""
    total = vehicle.m * state[3] * (derivative[6] + state[5])
    moment = vehicle.I * derivative[5]
    front = (moment + vehicle.lr * total) / (vehicle.lf + vehicle.lr)
    return np.array([front, total - front])


@pytest.mark.parametrize('sign', [-1, 1])
@pytest.mark.parametrize('acceleration', [-3.0, 0.0, 3.0])
@pytest.mark.parametrize('peak', [0.0, 0.4, 1.3])
def test_both_axles_saturate_at_load_dependent_limits(sign, acceleration, peak):
    vehicle = BASE.with_updates(mu_y_peak=peak)
    state = np.array([0, 0, 0, 5, 0, 0, -sign * 0.8], dtype=np.float32)
    result = derivatives(state, vehicle, acceleration)
    loads = vehicle.m / (vehicle.lf + vehicle.lr) * np.array([
        9.81 * vehicle.lr - acceleration * vehicle.h,
        9.81 * vehicle.lf + acceleration * vehicle.h,
    ])
    np.testing.assert_allclose(
        axle_forces(state, result, vehicle), sign * peak * loads,
        rtol=2e-6, atol=2e-6,
    )


def test_small_slip_preserves_linear_response_independently_of_peak():
    state = np.array([0, 0, 0.01, 5, 0, 0.02, 0.005], dtype=np.float32)
    low = derivatives(state, BASE.with_updates(mu_y_peak=0.5))
    high = derivatives(state, BASE.with_updates(mu_y_peak=2.0))
    np.testing.assert_array_equal(low, high)
    loads = BASE.m * 9.81 / (BASE.lf + BASE.lr) * np.array([BASE.lr, BASE.lf])
    slips = np.array([
        state[2] - state[6] - BASE.lf * state[5] / state[3],
        -state[6] + BASE.lr * state[5] / state[3],
    ])
    forces = BASE.mu * np.array([BASE.C_Sf, BASE.C_Sr]) * loads * slips
    np.testing.assert_allclose(axle_forces(state, low, BASE), forces, rtol=2e-6)


def test_front_and_rear_saturate_independently():
    state = np.array([0, 0, 0.4, 5, 0, 0, 0], dtype=np.float32)
    result = derivatives(state, BASE)
    forces = axle_forces(state, result, BASE)
    front_load = BASE.m * 9.81 * BASE.lr / (BASE.lf + BASE.lr)
    np.testing.assert_allclose(forces, [BASE.mu_y_peak * front_load, 0], atol=2e-6)


@pytest.mark.parametrize('acceleration', [-9.0, 9.0])
def test_unloaded_axles_have_zero_force(acceleration):
    vehicle = BASE.with_updates(h=1.0)
    state = np.array([0, 0, 0, 5, 0, 0, -0.8], dtype=np.float32)
    forces = axle_forces(state, derivatives(state, vehicle, acceleration), vehicle)
    unloaded = 0 if acceleration > 0 else 1
    assert forces[unloaded] == pytest.approx(0.0, abs=5e-6)
    assert forces[1 - unloaded] > 0


@pytest.mark.parametrize('speed', [-2.0, 0.0, 0.49])
def test_peak_does_not_change_low_speed_branch(speed):
    state = np.array([0, 0, 0.2, speed, 0, 0.1, 0.3], dtype=np.float32)
    np.testing.assert_array_equal(
        derivatives(state, BASE.with_updates(mu_y_peak=0.0)),
        derivatives(state, BASE.with_updates(mu_y_peak=2.0)),
    )


@pytest.mark.parametrize('peak', [-1.0, np.nan, np.inf])
def test_invalid_peak_rejected(peak):
    with pytest.raises(ValueError, match='mu_y_peak'):
        BASE.with_updates(mu_y_peak=peak)


def test_peak_is_traced_under_vmap_and_has_finite_gradients():
    params = DynamicsParams.from_vehicle_parameters(BASE)
    batched = jax.tree.map(lambda x: jnp.array([x, x]), params)
    batched = replace(batched, mu_y_peak=jnp.array([0.3, 0.6]))
    state = jnp.array([0., 0., 0.4, 5., 0., 0., 0.])
    output = jax.jit(jax.vmap(single_track))(
        jnp.stack([state, state]), jnp.zeros((2, 2)), batched
    )
    np.testing.assert_allclose(output[1, 5:], 2 * output[0, 5:], rtol=2e-6)
    gradient = jax.grad(lambda p: single_track(state, jnp.zeros(2), p)[5])(params)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(gradient))
    assert float(gradient.mu_y_peak) > 0


def test_peak_reaches_both_gym_environments_and_device_randomization():
    from f1tenth_gym.envs.collision_models import CollisionCheckMode
    from f1tenth_gym.envs.env_config import (
        DomainRandomizationConfig, EnvConfig, SimulationConfig,
    )
    from f1tenth_gym.envs.f110_env import F110Env
    from f1tenth_gym.envs.integrators import IntegratorType
    from f1tenth_gym.envs.jax_env import JaxF110Env
    from f1tenth_gym.envs.jax_simulator import JaxSimulator
    from f1tenth_gym.envs.lidar import LiDARConfig
    from f1tenth_gym.envs.track import Track

    theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    track = Track.from_refline(
        x=10 * np.cos(theta), y=10 * np.sin(theta), velx=np.full(64, 5.),
    )
    initial = np.array([[10, 0, 0.4, 5, 0, 0, 0]], dtype=np.float32)
    config = EnvConfig(
        map_name=track, params=BASE.with_updates(mu_y_peak=0.4),
        render_enabled=False, collision_check=CollisionCheckMode.NONE,
        lidar_config=LiDARConfig(enabled=False),
        simulation_config=SimulationConfig(
            integrator=IntegratorType.EULER, max_laps=None,
        ),
    )
    expected = initial[0] + config.simulation_config.timestep * vehicle_dynamics_st(
        initial[0], np.zeros(2), config.params.to_array(),
    )
    for env_cls in (F110Env, JaxF110Env):
        env = env_cls(config=config)
        try:
            env.reset(seed=3, options={'states': initial.copy()})
            obs, *_ = env.step(np.array([[0.4, 5.]], dtype=np.float32))
            np.testing.assert_allclose(obs['agent_0']['state'], expected, atol=2e-6)
        finally:
            env.close()

    randomized = config.with_updates(domain_randomization_config=DomainRandomizationConfig(
        enabled=True, low=BASE.with_updates(mu_y_peak=0.2),
        high=BASE.with_updates(mu_y_peak=0.8),
    ))
    simulator = JaxSimulator(randomized, track)
    _, batch = simulator.reset_batch(jax.random.split(jax.random.key(5), 4))
    peaks = np.asarray(batch.params.dynamics.vehicle.mu_y_peak)
    assert np.all((peaks >= 0.2) & (peaks <= 0.8))
    assert np.ptp(peaks) > 0
