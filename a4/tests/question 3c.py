from otter.test_files import test_case

OK_FORMAT = False

name = "question 3c"
points = 9

@test_case(points=2, hidden=False)


def test_init_type(
    gym,
    np,
    torch,
    functools,
    REINFORCEWithBaselinePolicy,
    policy_init_network,
    value_init_network,
):
    torch.manual_seed(0)
    np.random.seed(0)
    env = gym.make("CartPole-v0")
    test_state, _ = env.reset(seed=0)
    assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)
    test_policy = REINFORCEWithBaselinePolicy(
        env, policy_init_network(env), value_init_network(env), 0.99
    )
    assert_equal(isinstance(test_policy.opt, torch.optim.Optimizer), True)
    assert type(test_policy.value_opt).__name__ == "Adam"
    assert_equal(test_policy.value_opt.param_groups[0]["lr"], 2e-3)
    assert_equal(isinstance(test_policy.value_network, torch.nn.Sequential), True)


@test_case(points=1, hidden=False)


def test_update_type(
    gym,
    np,
    torch,
    functools,
    REINFORCEWithBaselinePolicy,
    policy_init_network,
    value_init_network,
    generate_episode,
):
    torch.manual_seed(0)
    np.random.seed(0)
    test_env = gym.make("CartPole-v0")
    assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)
    test_reinforce = REINFORCEWithBaselinePolicy(
        test_env, policy_init_network(test_env), value_init_network(test_env), 0.99
    )
    (
        test_states,
        test_actions,
        test_rewards,
        test_terminated,
        test_truncated,
    ) = generate_episode(test_env, test_reinforce)
    loss = test_reinforce.update(
        test_states, test_actions, test_rewards, test_terminated
    )
    assert_equal(isinstance(loss, dict), True)
    assert_equal("policy_loss" in loss, True)
    assert_equal("value_loss" in loss, True)
    assert_equal(isinstance(loss["policy_loss"], float), True)


@test_case(points=2, hidden=False)


def test_update_type_1(
    gym,
    np,
    torch,
    functools,
    REINFORCEWithBaselinePolicy,
    policy_init_network,
    value_init_network,
):
    torch.manual_seed(0)
    np.random.seed(0)
    test_env = gym.make("CartPole-v0")
    assert_equal = functools.partial(torch.testing.assert_close, rtol=1e-3, atol=0)
    test_reinforce = REINFORCEWithBaselinePolicy(
        test_env, policy_init_network(test_env), value_init_network(test_env), 0.99
    )
    test_states = list(np.random.random((10, 4)))
    test_actions = [0, 1, 0, 1, 0, 0, 0, 1, 1, 0]
    test_rewards = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    test_terminated = [False] * 10
    test_terminated[-1] = True
    loss = test_reinforce.update(
        test_states, test_actions, test_rewards, test_terminated
    )
    assert_equal(loss["policy_loss"], 4.115862846374512)
    assert_equal(loss["value_loss"], 36.25490188598633)


