"""
nn_layers.py — custom Keras layers of the surrogate networks (registered for keras.models.load_model)
===================================================================================================
15/10/2026  (plan ~/.claude/plans/floofy-purring-fiddle.md: jump-aware output head)

StepHead turns 2 x 7 raw numbers per curve into two jumps on the 256-point window axis of the network,
q = k / (n - 1), k = 0 .. n-1.  Each jump is a smooth step plus a localized spike, because the jumps of the
training curves usually carry a one-point overshoot right at the step (see nn-training/jump_labels/label_examples.png):

    jump_j(q) = g_j [ h_j sigmoid((q - u_j) / w_j) + a_j exp(-((q - u_j - d_j) / v_j)^2) ]

    j = 0 'end'      : the drop at the high-QFLS end of the window (t -> 0); u_0 is restricted to the last
                       `end_span` grid intervals: u_0 = 1 - (end_span / (n-1)) * sigmoid(raw)
    j = 1 'interior' : a jump anywhere, u_1 = sigmoid(raw)
    g = sigmoid (is there a jump?), h = height, w = (w_min + softplus) / (n-1)            (steps, supervised)
    a = spike amplitude, v = (w_min + softplus) / (n-1), d = 2 tanh / (n-1) (offset ≤ 2 points)  (learned freely)

A step adds h to every point AFTER u (higher index = higher QFLS); the smooth part of the network carries the rest.
Outputs: the summed jumps (B, n) and the step parameters (B, 2, 3) = (g, u, g*h) for the auxiliary supervision.
Import this module before loading a network that contains the layer (inference.surrogate.load_artifacts does).
"""
import keras
from keras import ops

N_RAW = 7          # raw numbers per jump: gate, height, location, width, spike amplitude, spike width, spike offset


@keras.saving.register_keras_serializable(package="trpl")
class StepHead(keras.layers.Layer):
    def __init__(self, n_points: int = 256, end_span: float = 8.0, w_min: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.n_points, self.end_span, self.w_min = int(n_points), float(end_span), float(w_min)

    def call(self, raw):                                   # raw (B, 2, N_RAW)
        n1 = float(self.n_points - 1)
        g = ops.sigmoid(raw[:, :, 0])
        h = raw[:, :, 1]
        u_end = 1.0 - (self.end_span / n1) * ops.sigmoid(raw[:, 0, 2])
        u_int = ops.sigmoid(raw[:, 1, 2])
        u = ops.stack([u_end, u_int], axis=1)
        w = (self.w_min + ops.softplus(raw[:, :, 3])) / n1
        a = raw[:, :, 4]
        v = (self.w_min + ops.softplus(raw[:, :, 5])) / n1
        d = 2.0 * ops.tanh(raw[:, :, 6]) / n1
        q = ops.linspace(0.0, 1.0, self.n_points)[None, None, :]
        step = h[:, :, None] * ops.sigmoid((q - u[:, :, None]) / w[:, :, None])
        spike = a[:, :, None] * ops.exp(-ops.square((q - u[:, :, None] - d[:, :, None]) / v[:, :, None]))
        jumps = ops.sum(g[:, :, None] * (step + spike), axis=1)
        return jumps, ops.stack([g, u, g * h], axis=2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_points), (input_shape[0], 2, 3)

    def get_config(self):
        return {**super().get_config(), "n_points": self.n_points, "end_span": self.end_span, "w_min": self.w_min}
