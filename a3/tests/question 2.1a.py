from otter.test_files import test_case

OK_FORMAT = False

name = "question 2.1a"
points = 5

@test_case(points=1, hidden=False)
def test_ev_mc_estimate_type(np, ev_mc_estimate):
    dummy_states = [0, 5, 0, 0, 0, 0, 5, 0, 5, 0, 0, 0, 5, 0, 0, 5, 5]
    dummy_actions = [1, 3, 0, 0, 3, 1, 3, 1, 3, 0, 0, 1, 0, 3, 1, 0]
    dummy_rewards = [-1, -1, -1, -1, -1, -1, -1, -10, -1, -100, -1, -1, -1, -1, -1, -1]

    predicted_answer = ev_mc_estimate(dummy_states, dummy_actions,
                                                dummy_rewards, discount=1.)

    np.testing.assert_allclose(len(predicted_answer.keys()), 2) # Only two distinct states
                                                                #are visited in the dummy episode
    np.testing.assert_allclose(len(predicted_answer[0]), 11) # State 0 is visited 11 ttimes

    # Make sure inner type is correct
    for k in predicted_answer:
        assert type(predicted_answer[k]) == list
