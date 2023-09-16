from otter.test_files import test_case

OK_FORMAT = False

name = "q1.a"
points = 5

@test_case(points=5, hidden=False)
def test_1a1_public(np, Bandit):
    dummy_bandit = Bandit(n_arm=3, actual_toxicity_prob=[0.0, 1, 0.0])
    test_arms = [0, 2, 1, 0, 0, 0, 2]
    for arm in test_arms:
        dummy_bandit.pull(arm)
    np.testing.assert_allclose(dummy_bandit.num_dose_selected[0],4)
    np.testing.assert_allclose(dummy_bandit.num_toxic[1],1)
