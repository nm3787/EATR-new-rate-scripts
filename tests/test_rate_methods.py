from __future__ import annotations

import random
import unittest
from importlib.util import find_spec

if find_spec("numpy") and find_spec("scipy"):
    import numpy as np
    import ks_censored as ksc
    import rate_methods_library as rm
else:
    np = None
    ksc = None
    rm = None


@unittest.skipUnless(np is not None and ksc is not None and rm is not None, "numpy and scipy are required")
class RateMethodTests(unittest.TestCase):
    def test_bootstrap_resamples_event_labels_with_sample(self):
        sample = ["a", "b", "c"]
        event = np.array([True, False, False])

        def collect(subsample, subevent):
            return sum(1 for item, flag in zip(subsample, subevent) if item == "a" and flag)

        random_state = random.getstate()
        random.seed(1)
        try:
            stats = rm.bootstrap(sample, collect, 5, event=event, return_stat=True)
        finally:
            random.setstate(random_state)

        self.assertEqual(len(stats), 5)
        self.assertTrue(all(value >= 0 for value in stats))

    def test_get_event_uses_shorter_trajectories_for_maxlen(self):
        data = [
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        ]

        event = rm.get_event(data, maxlen=3, quiet=True)

        self.assertEqual(event.tolist(), [True, False])

    def test_imetad_rescaled_times_uses_bias_column_when_acceleration_missing(self):
        data = [
            np.array([[2.0, 0.0, None, None], [4.0, 1.0, None, None]], dtype=object),
            np.array([[1.0, 0.5, None, None], [3.0, 0.5, None, None]], dtype=object),
        ]

        times = rm.iMetaD_rescaled_times(data, beta=1.0)

        expected = np.array(
            [
                4.0 * np.mean(np.exp(np.array([0.0, 1.0]))),
                3.0 * np.mean(np.exp(np.array([0.5, 0.5]))),
            ]
        )
        self.assertTrue(np.allclose(times, expected))

    def test_avg_max_bias_builds_monotonic_max_bias_profile(self):
        data = [
            np.array(
                [
                    [0.0, 1.0, None, None],
                    [1.0, 3.0, None, None],
                    [2.0, 2.0, None, None],
                ],
                dtype=object,
            ),
            np.array(
                [
                    [0.0, 2.0, None, None],
                    [1.0, 1.0, None, None],
                    [2.0, 4.0, None, None],
                ],
                dtype=object,
            ),
        ]

        averaged = rm.avg_max_bias(data, beta=1.0)

        self.assertTrue(np.allclose(averaged[:, 1], [1.5, 2.5, 3.5]))

    def test_prepare_times_sorts_and_aggregates_duplicates(self):
        t, n, d, l, m, transitions = ksc.prepare_times(
            [3.0, 1.0, 1.0, 2.0],
            [False, True, False, True],
        )

        self.assertTrue(np.allclose(t, [1.0, 2.0, 3.0]))
        self.assertEqual(n.tolist(), [4, 2, 1])
        self.assertEqual(d.tolist(), [1, 1, 0])
        self.assertEqual(l.tolist(), [1, 0, 1])
        self.assertEqual(m, 3)
        self.assertEqual(transitions.tolist(), [1.0, 2.0])
