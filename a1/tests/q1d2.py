from otter.test_files import test_case

OK_FORMAT = False

name = "q1.d2"
points = 5

@test_case(points=1, hidden=False)
def test_1d21_public(np, Bandit, boltzmann):
    dummy_bandit = Bandit(n_arm=3, n_pulls=1000, actual_toxicity_prob=[0, 1.0, 0])
    N=10
    for n in range(N):
        rew_rec_n, avg_ret_rec_n, tot_reg_rec_n = boltzmann(dummy_bandit, tau=1e8) # uniform sampling
        np.testing.assert_allclose(tot_reg_rec_n[-1],430, atol=20)