from otter.test_files import test_case

OK_FORMAT = False

name = "q1.f1"
points = 5

@test_case(points=1, hidden=False)
def test_1f11_public(np, Bandit, gradient_bandit):
    dummy_bandit = Bandit(n_arm=3, n_pulls=1000, actual_toxicity_prob=[0, 1.0, 0])
    N=20
    for n in range(N):
        dummy_bandit.init_bandit()
        rew_rec_n, avg_ret_rec_n, tot_reg_rec_n, opt_act_rec_n = gradient_bandit(dummy_bandit, alpha=0)
        np.testing.assert_allclose(tot_reg_rec_n[-1],135, atol=20)
