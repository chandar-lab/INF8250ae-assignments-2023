from otter.test_files import test_case

OK_FORMAT = False

name = "question 2.4b"
points = 5

@test_case(points=1, hidden=False)
def test_modified_tdn_type(np, FloorIsLava, make_eps_greedy_policy, modified_tdn):
    answers = {1: [-12.14359154, -15.00479634, -16.76559563, -18.03477267,
        -19.02451728, -19.83416103, -20.51819894, -21.10973223,
        -21.6303582 , -22.09493901, -22.51413286, -22.89583936,
        -23.24607351, -23.5695194 , -23.86989476, -24.15019889,
        -24.412886  , -24.65998929, -24.89321175, -25.11399375],
            10: [-50.34716411, -58.28996713, -62.60144165, -65.46553558,
        -67.5675138 , -69.2058335 , -70.5354492 , -71.64637001,
        -72.59515746, -73.41949171, -74.14563424, -74.79255955,
        -75.37438652, -75.90188156, -76.38342658, -76.82566406,
        -77.23393969, -77.61261395, -77.96528619, -78.29495916],
            45: [-96.29357908, -98.03394659, -98.64911857, -98.96608494,
        -99.16014544, -99.29150293, -99.3864821 , -99.4584462 ,
        -99.51491004, -99.56042854, -99.59792566, -99.62936565,
        -99.65611767, -99.67916568, -99.69923528, -99.71687348,
        -99.73250054, -99.74644477, -99.75896643, -99.77027451]}


    env = FloorIsLava(map_name="6x6", slip_rate=0.)
    dummy_state_action_values = np.arange(36*4).reshape(36,4)
    dummy_pi = make_eps_greedy_policy(dummy_state_action_values, 0.)


    for n in [1, 10, 45]:
        dummy_state_vals = modified_tdn(dummy_pi, env, n=n, num_episodes=20)
        np.testing.assert_allclose(np.array(answers[n]), np.array(dummy_state_vals)[:,0], atol=1e-2)
