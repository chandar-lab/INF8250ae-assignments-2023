from otter.test_files import test_case

OK_FORMAT = False

name = "q1.d1"
points = 5

@test_case(points=1, hidden=False)
def test_1d11_public(np, boltzmann_policy):
    N=1000
    dummy_x = [20, 100, 60]
    tmp_sum = []
    for n in range(N):
        dummy_idx = boltzmann_policy(dummy_x, tau=1e10) # uniform sampling
        tmp_sum.append(dummy_x[dummy_idx])
    np.testing.assert_allclose(np.mean(tmp_sum), 60, atol=5)