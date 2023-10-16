from otter.test_files import test_case

OK_FORMAT = False

name = "question 0.1a"
points = 2

@test_case(points=1, hidden=False)
def test_eps_greedy_policy_type(np, make_eps_greedy_policy):
    for epsilon in [0, 0.15]:
        dummy_state_action_values = np.ones((16, 4))
        dummy_state_action_values[:, 1] = 10
        test_pi = make_eps_greedy_policy(dummy_state_action_values, epsilon)
        states = list(range(16)) * 10000
        all_actions = []
        for s in states:
            action = test_pi(s)
            all_actions.append(action)
        expected_rate = (1-epsilon + epsilon/4)
        np.testing.assert_allclose(np.sum(np.array(all_actions) == 1)/len(states), expected_rate, atol=0.015)
