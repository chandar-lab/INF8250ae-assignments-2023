from otter.test_files import test_case

OK_FORMAT = False

name = "question 4.1a"
points = 15

@test_case(points=1, hidden=False)
def test_get_action_shape_type(np, gym, DQNAgent):
    env = gym.make("CartPole-v1")
    agent = DQNAgent(env.observation_space, env.action_space, 0.0, learning_starts_at=0,
        learning_frequency=1,
        learning_rate=.01,
        discount_factor=.9,
        buffer_size=100,
        target_update_frequency=100,
        batch_size=128)
    state, _ = env.reset()
    action = agent.get_action(state)
    assert isinstance(action, int) or isinstance(action, np.int64) or isinstance(action, np.int32)
    assert env.action_space.contains(action)
    agent = DQNAgent(env.observation_space, env.action_space, 1.0, learning_starts_at=0,
        learning_frequency=1,
        learning_rate=.01,
        discount_factor=.9,
        buffer_size=100,
        target_update_frequency=100,
        batch_size=128)
    action = agent.get_action(state)
    assert isinstance(action, int) or isinstance(action, np.int64) or isinstance(action, np.int32)
    assert env.action_space.contains(action)

@test_case(points=2, hidden=False)
def test_compute_targets_shape_type(torch, gym, DQNAgent):
    env = gym.make("CartPole-v1")
    agent = DQNAgent(env.observation_space, env.action_space, 0.0, learning_starts_at=0,
        learning_frequency=1,
        learning_rate=.01,
        discount_factor=.99,
        buffer_size=100,
        target_update_frequency=100,
        batch_size=128)
    rewards = torch.randint(0, 10, (128,))
    next_observations = torch.rand((128, *env.observation_space.shape))
    terminated = torch.randint(0, 2, (128,))
    targets = agent.compute_targets(rewards, next_observations, terminated)
    assert isinstance(targets, torch.Tensor)
    assert targets.shape == (128,)

