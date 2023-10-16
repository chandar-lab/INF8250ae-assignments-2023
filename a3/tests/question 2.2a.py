from otter.test_files import test_case

OK_FORMAT = False

name = "question 2.2a"
points = 5

@test_case(points=1, hidden=False)
def test_td0_type(np, FloorIsLava, make_eps_greedy_policy, td0):
    answer = [ -1.49378919,  -2.96526431,  -4.41475871,  -5.84260071,
            -7.24911376,  -8.63461647,  -9.99942269, -11.34384158,
        -12.66817769, -13.97273101]

    env = FloorIsLava(map_name="6x6", slip_rate=0.)

    dummy_state_action_values = np.arange(36*4).reshape(36,4)
    dummy_pi = make_eps_greedy_policy(dummy_state_action_values, 0.)

    dummy_state_vals = td0(dummy_pi, env, step_size=0.01, num_episodes=10)

    np.testing.assert_allclose(np.array(answer), np.array(dummy_state_vals)[:, 0])

