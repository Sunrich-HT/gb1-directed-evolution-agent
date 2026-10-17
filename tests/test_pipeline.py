"""Protocol tests: the properties that make the comparison trustworthy."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gb1agent.agent import Hypothesis, judge_hypotheses, parse_response
from gb1agent.features import onehot, pairwise
from gb1agent.metrics import top_percentile_precision, topk_recall
from gb1agent.priors import zscore_by_substitution


class TestEncodings(unittest.TestCase):
    def test_shapes_and_counts(self):
        X = onehot(["VDGV", "FWAA"])
        self.assertEqual(X.shape, (2, 80))
        self.assertTrue(np.allclose(X.sum(axis=1), 4))
        P = pairwise(["VDGV"])
        self.assertEqual(P.shape, (1, 2480))
        self.assertEqual(int(P.sum()), 10)   # 4 additive + 6 pairwise terms


class TestMetrics(unittest.TestCase):
    def test_perfect_and_inverted(self):
        y = np.arange(100.0)
        self.assertEqual(topk_recall(y, y, k=10), 1.0)
        self.assertEqual(topk_recall(y, -y, k=10), 0.0)
        self.assertEqual(top_percentile_precision(y, y, pct=10.0), 1.0)


class TestAgentContract(unittest.TestCase):
    def test_parser_rejects_illegal_ids(self):
        legal = {"VDGV", "FWAA"}
        sel, hyps, _ = parse_response(
            '{"selection": ["VDGV", "ZZZZ", "VDGV", "FWAA"], "hypotheses": []}', legal, 4)
        self.assertEqual(sel, ["VDGV", "FWAA"])     # illegal and duplicate dropped
        self.assertEqual(hyps, [])

    def test_hypothesis_verdicts_are_program_assigned(self):
        variants = ["FDGV", "FDGA", "FDGT", "VDGV", "VDGA", "LDGV", "LDGA"]
        fitness = [5.0, 4.0, 4.5, 1.0, 1.0, 1.0, 1.0]
        h = Hypothesis(statement="F at 39 helps", site=39, substitution="V39F",
                       direction="beneficial")
        judged = judge_hypotheses([h], variants, fitness, incumbent=1.0)
        self.assertEqual(judged[0].verdict, "supported")
        h2 = Hypothesis(statement="F at 39 hurts", site=39, substitution="V39F",
                        direction="deleterious")
        self.assertEqual(judge_hypotheses([h2], variants, fitness, 1.0)[0].verdict, "refuted")

    def test_undetermined_when_batch_lacks_contrast(self):
        h = Hypothesis(statement="W at 40", site=40, substitution="D40W",
                       direction="beneficial")
        judged = judge_hypotheses([h], ["VWGV", "VDGV"], [3.0, 1.0], 1.0)
        self.assertEqual(judged[0].verdict, "undetermined")


class TestPriors(unittest.TestCase):
    def test_zscore_removes_substitution_bias(self):
        table = np.array([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [4.0, 8.0]])
        z = zscore_by_substitution(table)
        self.assertTrue(np.allclose(z[:, 0], z[:, 1]))   # constant offset removed


class TestSplit(unittest.TestCase):
    def test_pool_can_contain_the_optimum(self):
        """The defect this project was rebuilt to avoid."""
        from gb1agent.data import make_split
        import pandas as pd

        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "variant": [f"V{i:03d}" for i in range(2000)],
            "fitness": rng.random(2000) * 9,
            "n_mut": 4,
            "sequence": "x",
        })
        sp = make_split(df, seed=1, n_test=200, n_initial=50)
        pool_max = df["fitness"].to_numpy()[sp.pool].max()
        start_max = df["fitness"].to_numpy()[sp.initial].max()
        self.assertGreater(pool_max, start_max)


if __name__ == "__main__":
    unittest.main()
