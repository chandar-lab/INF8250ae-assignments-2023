from otter.test_files import test_case

OK_FORMAT = False

name = "question 3.3a"
points = 5

@test_case(points=1, hidden=False)
def test_agent_step_type(np, QLearningAgent, td_control):
    answer = [-119, -122, -115, -118, -121, -125, -106, -116, -117, -110]
    returns, agent = td_control(agent_class=QLearningAgent, epsilon=0, step_size=0.1,
            run=0, num_episodes=10, discount=1.)
    np.testing.assert_allclose(answer, returns)
