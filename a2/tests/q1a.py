from otter.test_files import test_case

OK_FORMAT = False

name = "q1a"
points = 4

@test_case(points=2, hidden=False)
def test_evaluate_agent_public(gym, np, ExpertAgent, evaluate_agent):
    env = gym.make("Ant-v4")
    env.reset(seed=42)
    expert_1mil = ExpertAgent("experts/network_1mil.pt")
    mean, std = evaluate_agent(expert_1mil, env, 10)
    assert mean > 4000, "The mean return of the expert should be greater than 4000"
    assert std != 0, "The standard deviation of the expert should not be 0"
