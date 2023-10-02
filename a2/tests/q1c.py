from otter.test_files import test_case

OK_FORMAT = False

name = "q1c"
points = 3

@test_case(points=1, hidden=False)
def test_agent_forward(torch, Agent):
    agent = Agent(20, 6)
    obs = torch.randn(4, 20)
    action = agent(obs)
    assert type(action) == torch.Tensor
    assert action.shape == (4, 6)
    assert torch.allclose(action, agent._network(obs))
    
