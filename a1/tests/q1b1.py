from otter.test_files import test_case

OK_FORMAT = False

name = "q1.b1"
points = 5

@test_case(points=1, hidden=False)
def test_1b1_public(np, Bandit, eps_greedy):
    N=20
    dummy_bandit = Bandit(n_arm=3, n_pulls=1000, actual_toxicity_prob=[1, 0.0, 0.0])
    for n in range(N):
        dummy_bandit.init_bandit()
        rew_rec_n, avg_ret_rec_n, tot_reg_rec_n, opt_act_rec_n = eps_greedy(dummy_bandit, eps=1)
        np.testing.assert_allclose(tot_reg_rec_n[-1],730,atol=20)