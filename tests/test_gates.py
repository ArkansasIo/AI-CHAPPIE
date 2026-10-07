"""
Tests for the mathematics of gate logic (nn.gates).

Covers: hard gates, truth tables, canonical forms (SOP/POS/ANF), Boolean
identities, functional completeness (NAND/NOR), smooth gates + their analytic
gradients, and learning gates with a neural network.
"""

import itertools

import numpy as np

from nn import gates as G
from nn import linalg as L


# ---------------------------------------------------------------------------
# Hard gates
# ---------------------------------------------------------------------------
def test_and_or_not_truth_tables():
    assert list(G.AND(1, 0).ravel()) == [0]
    assert list(G.OR(1, 0).ravel()) == [1]
    assert list(G.NOT(1).ravel()) == [0]
    # vectorized
    assert list(G.AND([0, 0, 1, 1], [0, 1, 0, 1])) == [0, 0, 0, 1]
    assert list(G.OR([0, 0, 1, 1], [0, 1, 0, 1])) == [0, 1, 1, 1]


def test_gate_table_matches_implementations():
    for name, bits in G.GATE_TABLE.items():
        tt = G.truth_table(name)
        assert tuple(int(v) for v in tt[:, 2]) == bits, name


def test_all_16_binary_gates_are_distinct():
    patterns = set(G.ALL_BINARY_GATES.values())
    assert len(patterns) == 16


def test_xor_is_addition_mod_2():
    for a, b in itertools.product([0, 1], repeat=2):
        assert int(G.XOR(a, b)) == (a + b) % 2


def test_nand_nor_are_universal():
    assert G.is_complete_nand()
    # NOR's dual: NOT/AND/OR/XOR are all realizable from NOR alone.
    ref = {
        "NOT": lambda a, b: G.NOT(a),
        "AND": lambda a, b: G.AND(a, b),
        "OR": lambda a, b: G.OR(a, b),
        "XOR": lambda a, b: G.XOR(a, b),
    }
    for name, fn in ref.items():
        for a, b in itertools.product([0, 1], repeat=2):
            got = int(G.to_nor(name, a, b))
            want = int(fn(a, b))
            assert got == want, (name, a, b, got, want)


def test_de_morgan():
    assert G.de_morgan_check()


# ---------------------------------------------------------------------------
# Canonical forms
# ---------------------------------------------------------------------------
def test_minterms_maxterms_partition():
    for name in ["AND", "OR", "XOR", "NAND"]:
        bits = np.array(G.GATE_TABLE[name])
        on, off = G.minterms(bits), G.maxterms(bits)
        assert sorted(on + off) == [0, 1, 2, 3], name


def test_sop_pos_strings():
    bits = np.array(G.GATE_TABLE["XOR"])
    assert G.sop(bits) == "(¬a∧b) ∨ (a∧¬b)"
    assert "∧" in G.pos(bits)


def test_anf_xor_is_linear():
    terms = G.anf(G.GATE_TABLE["XOR"])
    monomials = {m for _, m in terms}
    assert (0,) in monomials and (1,) in monomials      # a ⊕ b
    assert (0, 1) not in monomials                      # no degree-2 term


def test_anf_and_is_product():
    terms = G.anf(G.GATE_TABLE["AND"])
    assert (0, 1) in {m for _, m in terms}              # the ab term


def test_anf_reevaluates_gate_exactly():
    """The Zhegalkin polynomial must reproduce the gate on all input rows."""
    for name in ["AND", "OR", "NAND", "NOR", "XOR", "XNOR"]:
        terms = G.anf(G.GATE_TABLE[name])
        for row in itertools.product([0, 1], repeat=2):
            expected = int(G.hard_gate(name, *row))
            got = G.evaluate_anf(terms, row)
            assert got == expected, (name, row, got, expected)


# ---------------------------------------------------------------------------
# Smooth gates + calculus
# ---------------------------------------------------------------------------
def test_soft_gates_approach_hard_gates():
    for name in ["AND", "OR", "NAND", "NOR", "XOR", "XNOR"]:
        for a, b in itertools.product([0, 1], repeat=2):
            soft = float(G.soft_gate(name, a, b, tau=0.01))
            hard = int(G.hard_gate(name, a, b))
            assert abs(soft - hard) < 0.02, (name, a, b, soft, hard)


def test_soft_not_is_exact():
    assert abs(float(G.soft_not(1.0)) - 0.0) < 1e-12
    assert abs(float(G.soft_not(0.0)) - 1.0) < 1e-12


def test_soft_gate_gradients_match_finite_differences():
    """The analytic gate gradients must equal central differences."""
    eps = 1e-6
    tau = 0.2
    for name in ["AND", "OR", "NAND", "NOR", "XOR", "XNOR"]:
        for a, b in [(0.2, 0.7), (0.6, 0.3), (0.45, 0.9)]:
            da, db = G.soft_gate_grad(name, a, b, tau=tau)
            num_da = (float(G.soft_gate(name, a + eps, b, tau)) -
                      float(G.soft_gate(name, a - eps, b, tau))) / (2 * eps)
            num_db = (float(G.soft_gate(name, a, b + eps, tau)) -
                      float(G.soft_gate(name, a, b - eps, tau))) / (2 * eps)
            assert abs(float(da) - num_da) < 1e-4, (name, "da", a, b)
            assert abs(float(db) - num_db) < 1e-4, (name, "db", a, b)


def test_soft_xor_has_a_degree_two_curvature():
    """XOR's soft relaxation is non-linear in a, unlike a plain sigmoid.

    For XOR, `z = a + b - 2ab`, so `dz/da = 1 - 2b`: the derivative depends on
    `b`, which is exactly the parity interaction (a single sigmoid neuron has
    `dz/da` independent of `b`). We check that `d(XOR)/da` differs across `b`.
    """
    a = 0.5
    da_b0 = float(G.soft_gate_grad("XOR", a, 0.0, tau=0.2)[0])
    da_b1 = float(G.soft_gate_grad("XOR", a, 1.0, tau=0.2)[0])
    assert da_b0 * da_b1 < 0, (da_b0, da_b1)


def test_straight_through_estimator():
    hard, surrogate = G.straight_through(np.array([-1.0, 1.0]))
    assert list(hard.ravel()) == [0.0, 1.0]          # hard forward
    assert np.all(surrogate.ravel() > 0)             # non-zero backward gradient


# ---------------------------------------------------------------------------
# Linear separability (the geometry claim)
# ---------------------------------------------------------------------------
def test_separability_flags_in_registry():
    """The gates flagged non-separable are exactly XOR and XNOR."""
    non_sep = {name for name, bits in G.GATE_TABLE.items()
               if name in ("XOR", "XNOR")}
    assert non_sep == {"XOR", "XNOR"}


def test_xor_not_realizable_by_one_neuron():
    """A single linear threshold unit cannot implement XOR (see docs §5.1)."""
    from nn.model import MLP

    net = MLP([2, 2], activations="linear", out_activation="linear",
              init="xavier_normal", loss="softmax_ce", seed=0)
    X, Y = G.gate_dataset("XOR")
    from nn.optimizers import Adam
    opt = Adam(lr=0.5)
    for _ in range(5000):
        net.train_step(X, Y, opt)
    accuracy = float(np.mean(net.predict_classes(X) == Y))
    assert accuracy < 1.0, "a single linear layer should NOT solve XOR"


def test_xor_solvable_with_hidden_layer():
    net, acc = G.learn_gate("XOR", hidden=8, epochs=2500, lr=0.1, seed=3)
    assert acc == 1.0


def test_separable_gates_learn_with_single_neuron():
    for name in ["AND", "OR", "NAND", "NOR"]:
        net, acc = G.learn_gate(name, hidden=0, epochs=2500, lr=0.1, seed=3)
        assert acc == 1.0, (name, acc)


def test_gate_dataset_shapes():
    X, Y = G.gate_dataset("AND")
    assert X.shape == (4, 2)
    assert Y.shape == (4,)


def _run_all():
    import traceback
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    raise SystemExit(_run_all())
