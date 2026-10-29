"""Tests of inference/nn_layers.py (StepHead of the jump-aware surrogate head)."""
import numpy as np


def _raw(gate_logit, height, loc_logit, width_raw, spike=(0.0, 0.0)):
    from inference.nn_layers import N_RAW
    r = np.zeros((1, 2, N_RAW), np.float32)
    r[0, :, 0], r[0, :, 1], r[0, :, 2], r[0, :, 3], r[0, :, 4] = gate_logit, height, loc_logit, width_raw, spike
    r[0, :, 5] = -20.0                                                       # sharpest spike width
    return r


def test_step_location_height_and_shapes():
    import keras
    from inference.nn_layers import StepHead
    layer = StepHead(n_points=256, end_span=8.0, w_min=0.3)
    raw = _raw([20.0, 20.0], [-0.5, 0.2], [0.0, 0.0], [-20.0, -20.0])   # gates open, sharpest width
    steps, params = layer(keras.ops.convert_to_tensor(raw))
    steps, params = np.asarray(steps), np.asarray(params)
    assert steps.shape == (1, 256) and params.shape == (1, 2, 3)
    # interior step at u = 0.5 -> between points 127 and 128, height +0.2; end step 4 intervals before the end
    assert np.isclose(steps[0, 120], 0.0, atol=1e-4) and np.isclose(steps[0, 135], 0.2, atol=1e-4)
    u_end = 1 - 8 / 255 * 0.5
    assert np.isclose(params[0, 0, 1], u_end, atol=1e-6)
    assert np.isclose(steps[0, 255], 0.2 - 0.5, atol=1e-3)                   # both steps active at the end
    assert np.isclose(params[0, 1, 2], 0.2, atol=1e-6)                        # g*h of the interior step


def test_closed_gate_gives_no_step():
    import keras
    from inference.nn_layers import StepHead
    steps, _ = StepHead()(keras.ops.convert_to_tensor(_raw([-30.0, -30.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0])))
    assert np.abs(np.asarray(steps)).max() < 1e-6


def test_save_load_round_trip(tmp_path):
    import keras
    from inference.nn_layers import StepHead
    inp = keras.Input((8,))
    from inference.nn_layers import N_RAW
    raw = keras.layers.Reshape((2, N_RAW))(keras.layers.Dense(2 * N_RAW)(inp))
    steps, params = StepHead(name="steps")(raw)
    m = keras.Model(inp, steps)
    x = np.random.default_rng(0).normal(size=(4, 8)).astype(np.float32)
    m.save(tmp_path / "m.keras")
    m2 = keras.models.load_model(tmp_path / "m.keras", compile=False)
    assert np.allclose(m.predict(x, verbose=0), m2.predict(x, verbose=0), atol=1e-6)


def test_spike_is_local():
    import keras
    from inference.nn_layers import StepHead
    raw = _raw([20.0, -30.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], spike=(0.7, 0.0))    # end jump: spike only
    steps, params = StepHead()(keras.ops.convert_to_tensor(raw))
    s = np.asarray(steps)[0]
    k = int(np.argmax(s))
    assert k >= 247 and np.isclose(s[k], 0.7, atol=0.1)
    assert np.abs(s[:240]).max() < 1e-6                                         # nothing far from the jump
