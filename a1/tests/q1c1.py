from otter.test_files import test_case

OK_FORMAT = False

name = "q1.c1"
points = 5

@test_case(points=1, hidden=False)
def test_1c11_public(np, Bandit, ucb):
    dummy_bandit = Bandit(n_arm=3, n_pulls=1000, actual_toxicity_prob=[0, 1.0, 0])
    N=20
    for n in range(N):
        rew_rec_n, avg_ret_rec_n, tot_reg_rec_n, opt_act_rec_n = ucb(dummy_bandit, c=10)
        np.testing.assert_allclose(tot_reg_rec_n[-1],700, atol=30)