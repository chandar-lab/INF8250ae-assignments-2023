from otter.test_files import test_case

OK_FORMAT = False

name = "question 1b"
points = 12

@test_case(points=2, hidden=False)


def test_init_type(gym, np, torch, functools, Policy, policy_init_network):
    torch.manual_seed(0)
    np.random.seed(0)
    env = gym.make("CartPole-v1")
    test_state, _ = env.reset(seed=0)
    assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)
    test_policy = Policy(env, policy_init_network(env), 0.99)
    assert_equal(isinstance(test_policy.opt, torch.optim.Optimizer), True)
    assert type(test_policy.opt).__name__ == "Adam"
    assert_equal(test_policy.opt.param_groups[0]["lr"], 1e-3)


@test_case(points=1, hidden=False)


def test_dist_type(gym, np, torch, functools, Policy, policy_init_network):
    torch.manual_seed(0)
    np.random.seed(0)
    env = gym.make("CartPole-v1")
    test_state, _ = env.reset(seed=0)
    assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)
    test_policy = Policy(env, policy_init_network(env), 0.99)
    assert_equal(
        np.array(test_policy.distribution(test_state).param_shape), np.array([1, 2])
    )


@test_case(points=2, hidden=False)


def test_dist_type_1(gym, np, torch, functools, Policy, policy_init_network):
    torch.manual_seed(0)
    np.random.seed(0)
    env = gym.make("CartPole-v1")
    test_state = [-0.04267925, 0.01395816, 0.03371909, 0.04327]
    assert_equal = functools.partial(torch.testing.assert_close, rtol=1e-5, atol=0)
    test_policy = Policy(env, policy_init_network(env), 0.99)
    assert_equal(
        test_policy.distribution(test_state)
        .log_prob(torch.tensor([0, 1]))
        .detach()
        .numpy(),
        np.array([-0.87657535, -0.5382068]).astype(np.float32),
    )


@test_case(points=1, hidden=False)


def test_action_type(gym, np, torch, functools, Policy, policy_init_network):
    torch.manual_seed(0)
    np.random.seed(0)
    env = gym.make("CartPole-v1")
    test_state, _ = env.reset(seed=0)
    assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)
    test_policy = Policy(env, policy_init_network(env), 0.99)
    assert_equal(isinstance(test_policy.action(test_state), int), True)


@test_case(points=2, hidden=False)


def test_action_type_1(gym, np, torch, functools, Policy, policy_init_network):
    torch.manual_seed(0)
    np.random.seed(0)
    test_state = [0.925, 1.0, 0.22, 0.4]
    env = gym.make("CartPole-v1")
    network = policy_init_network(env)
    test_policy = Policy(env, network, 0.99)
    probs = network(
        torch.tensor(test_state).float().reshape(-1, env.observation_space.shape[0])
    ).flatten()
    probs = probs / probs.sum()
    N = 10000
    actions_list = []
    for i in range(N):
        action = test_policy.action(test_state)
        assert env.action_space.contains(
            action
        ), "Your action must be in the action space"
        actions_list.append(action)
    _, counts = np.unique(np.array(actions_list), return_counts=True)
    empirical_probs = (counts / N).astype(np.float32)
    torch.testing.assert_close(
        probs.detach().cpu().numpy(), empirical_probs, atol=0.015, rtol=0.01
    )


