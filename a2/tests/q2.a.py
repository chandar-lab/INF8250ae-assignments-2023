from otter.test_files import test_case

OK_FORMAT = False

name = "q2.a"
points = 15

@test_case(points=5, hidden=False)
def test_correct_loss(np, ReplayBuffer, behavior_cloning, torch):
    from unittest.mock import Mock, MagicMock, patch
    # agent = lambda x: x + torch.nn.parameter.Parameter(torch.Tensor([2., 2.]))
    # agent = MagicMock()
    class MockAgent():
        def __call__(self, x):
            return x + torch.nn.parameter.Parameter(torch.Tensor([2., 2.]))
        def forward(self, x):
            return x + torch.nn.parameter.Parameter(torch.Tensor([2., 2.]))
    agent = MockAgent()
    optimizer = Mock()
    buffer = ReplayBuffer()
    buffer.add_rollouts(
        {
            "states": np.arange(10).reshape(5, 2),
            "actions": np.arange(10).reshape(5, 2),
        }
    )
    loss = behavior_cloning(agent, optimizer, buffer, batch_size=5, steps=10)
    assert np.allclose(loss, 4.0)
    
