from otter.test_files import test_case

OK_FORMAT = False

name = "question 1.1a"
points = 5

@test_case(points=1, hidden=False)
def test_fv_mc_estimation_type(np, fv_mc_estimation):
    dummy_states = [0, 5, 11]
    dummy_actions = [1, 3]
    dummy_rewards = [-1, -100]

    true_answer = {(0, 1): -101.0,
    (5, 3): -100.0,
    }

    visited_states_returns = fv_mc_estimation(dummy_states, dummy_actions, dummy_rewards, discount=1.)
    for sa in true_answer:
        assert sa in visited_states_returns
        np.testing.assert_allclose(true_answer[sa], visited_states_returns[sa])
    assert len(true_answer) == len(visited_states_returns)
