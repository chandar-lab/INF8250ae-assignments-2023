from otter.test_files import test_case

OK_FORMAT = False

name = "q3"
points = 20

@test_case(points=3, hidden=False)
def test_relabel_trajectories_public(np, ExpertAgent, relabel_with_expert):
    expert = ExpertAgent("./public/a2/experts/network_1mil.pt")
    states = np.random.randn(10, 27)
    actions = relabel_with_expert(states, expert)
    assert actions.shape == (10, 8)

@test_case(points=2, hidden=False)
def test_collect_rollouts_public(gym, collect_rollouts, ExpertAgent):
    env = gym.make("Ant-v4")
    expert = ExpertAgent("./public/a2/experts/network_1mil.pt")
    states, actions = collect_rollouts(env, expert, 1000)
    assert states.shape == (1000, 27)
    assert actions.shape == (1000, 8)

@test_case(points=2, hidden=False)
def test_seed_data_public(gym, seed_data, ExpertAgent):
    from unittest.mock import Mock
    env = gym.make("Ant-v4")
    buffer = Mock()
    seed_data(env, ExpertAgent("./public/a2/experts/network_1mil.pt"), buffer, 1000)
    assert buffer.add_rollouts.call_count == 1
    assert buffer.add_rollouts.call_args[0][0]["states"].shape == (1000, 27)
    assert buffer.add_rollouts.call_args[0][0]["actions"].shape == (1000, 8)    


