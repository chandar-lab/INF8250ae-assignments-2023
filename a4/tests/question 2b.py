from otter.test_files import test_case

OK_FORMAT = False

name = "question 2b"
points = 7

@test_case(points=1, hidden=False)
def test_update_type(gym, np, torch, functools, REINFORCEPolicy, policy_init_network, generate_episode):
    torch.manual_seed(0)
    np.random.seed(0)
    test_env = gym.make('CartPole-v1')
    assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)
    test_reinforce = REINFORCEPolicy(test_env, policy_init_network(test_env), 0.99)
    test_states, test_actions, test_rewards, test_terminated, test_truncated = generate_episode(test_env, test_reinforce)
    loss = test_reinforce.update(test_states, test_actions, test_rewards, test_terminated)
    assert_equal(isinstance(loss,dict), True)
    assert_equal('policy_loss' in loss, True)
    assert_equal(isinstance(loss['policy_loss'], float), True)

@test_case(points=2, hidden=False)
def test_update_type_1(gym, np, torch, functools, REINFORCEPolicy, policy_init_network):

    torch.manual_seed(0)
    np.random.seed(0)
    test_env = gym.make('CartPole-v0')
    assert_equal = functools.partial(torch.testing.assert_close, rtol=1e-3, atol=0)
    test_reinforce = REINFORCEPolicy(test_env, policy_init_network(test_env), 0.99)
    test_states = list(np.random.random((10,4)))
    test_actions = [0, 1, 0, 1, 0, 0, 0, 1, 1, 0]
    test_rewards = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
    test_terminated = [False] * 10
    test_terminated[-1] = True
    loss = test_reinforce.update(test_states, test_actions, test_rewards, test_terminated)
    assert_equal(loss['policy_loss'], -4.09295129776001)

