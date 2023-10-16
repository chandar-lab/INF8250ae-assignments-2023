from otter.test_files import test_case

OK_FORMAT = False

name = "question 0.1b"
points = 3

@test_case(points=1, hidden=False)
def generate_episode_type(np, FloorIsLava, make_eps_greedy_policy, generate_episode):
    true_states = np.zeros(52)
    true_actions = np.ones(51) *3
    true_rewards =  np.ones(51)* -1
    true_rewards[-1] = -100

    env = FloorIsLava(map_name="4x4", slip_rate=0)
    dummy_state_action_values = np.arange(16*4).reshape(16,4)
    dummy_pi = make_eps_greedy_policy(dummy_state_action_values, 0.)
    states, actions, rewards = generate_episode(dummy_pi, env)

    np.testing.assert_allclose(len(states), len(true_states))
    np.testing.assert_allclose(states, true_states)
    np.testing.assert_allclose(actions, true_actions)
    np.testing.assert_allclose(rewards, true_rewards)
