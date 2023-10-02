from otter.test_files import test_case

OK_FORMAT = False

name = "q1b"
points = 3

@test_case(points=1, hidden=False)
def test_rb_public(np, ReplayBuffer):
    rb = ReplayBuffer()
    rb.add_rollouts(
        {
            "states": np.arange(50).reshape(25, 2),
            "actions": np.arange(50).reshape(25, 2),
        }
    )
    for _ in range(5):
        states, actions = rb.sample(10)
        assert states.shape == (10, 2)
        assert actions.shape == (10, 2)


