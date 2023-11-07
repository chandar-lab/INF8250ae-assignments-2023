from otter.test_files import test_case

OK_FORMAT = False

name = "question 4b"
points = 9

@test_case(points=2, hidden=False)
def test_init_type(gym, np, torch, functools, ActorCriticPolicy, policy_init_network, value_init_network):
    torch.manual_seed(0)
    np.random.seed(0)
    env = gym.make('CartPole-v0')
    test_state, _ = env.reset(seed = 0)
    assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)
    test_policy = ActorCriticPolicy(env, policy_init_network(env), value_init_network(env), 0.99)
    assert_equal(isinstance(test_policy.opt, torch.optim.Optimizer), True)
    assert type(test_policy.value_opt).__name__ == 'Adam'
    assert_equal(test_policy.value_opt.param_groups[0]['lr'], 2e-3)
    assert_equal(isinstance(test_policy.value_network, torch.nn.Sequential), True)
@test_case(points=1, hidden=False)
def test_train_epsiode_type(gym, np, torch, functools, ActorCriticPolicy, policy_init_network, value_init_network):
    torch.manual_seed(0)
    np.random.seed(0)
    test_env = gym.make('CartPole-v1')
    assert_equal = functools.partial(torch.testing.assert_close, rtol=1e-3, atol=0)
    test_actor_critic = ActorCriticPolicy(test_env, policy_init_network(test_env), value_init_network(test_env), 0.99)
    loss = test_actor_critic.train_episode()
    assert_equal(isinstance(loss,dict), True)
    assert_equal('policy_loss' in loss, True)
    assert_equal('value_loss' in loss, True)

@test_case(points=2, hidden=False)
def test_train_epsiode_type_1(gym, np, torch, functools, ActorCriticPolicy, policy_init_network, value_init_network):
    torch.manual_seed(0)
    np.random.seed(0)
    test_env = gym.make('CartPole-v1')
    assert_equal = functools.partial(torch.testing.assert_close, rtol=1e-3, atol=0)
    test_actor_critic = ActorCriticPolicy(test_env, policy_init_network(test_env), value_init_network(test_env), 1.)
    loss = test_actor_critic.train_episode()
    assert_equal(loss['policy_loss'],0.6701309680938721)
    assert_equal(loss['value_loss'], 0.4815424680709839)

