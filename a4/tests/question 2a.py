from otter.test_files import test_case

OK_FORMAT = False

name = "question 2a"
points = 5

@test_case(points=1, hidden=False)
def test_discounted_returns_type(gym, np, torch, functools, discounted_returns):
    test_len_rollout = 5
    test_done = np.array([True for _ in range(test_len_rollout)])
    test_r = np.ones(test_len_rollout)
    assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)
    test_ret = discounted_returns(test_r, test_done, 0.99)
    assert_equal(test_ret.shape[0], test_len_rollout)
@test_case(points=2, hidden=False)
def test_discounted_returns_type_1(gym, np, torch, functools, discounted_returns):
    assert_equal = functools.partial(torch.testing.assert_close, rtol=1e-5, atol=0)
    test_rewards = [1.0,2.0,3.0,4.0,2.0,1.0]
    test_done = [False, False, True, False, False, True]
    test_return = discounted_returns(test_rewards, test_done, 0.5)
    actual_return = np.array([2.75, 3.5, 3.0, 5.25, 2.5, 1.0])
    assert_equal(test_return, actual_return)
