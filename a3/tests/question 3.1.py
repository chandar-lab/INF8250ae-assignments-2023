from otter.test_files import test_case

OK_FORMAT = False

name = "question 3.1"
points = 3

@test_case(points=1, hidden=False)
def test_train_episode_type(np, FloorIsLava, train_episode, Agent):
    class DummyAgent(Agent):
        def agent_step(self, prev_state, prev_action, prev_reward, current_state, done):
            """ A learning step for the agent given SARS
            Args:
                prev_state (int): the state observation from the enviromnents last step
                prev_action (int): the action taken given prev_state
                prev_reward (float): The reward received for taking prev_action in prev_state
                current_state (int): The state received for taking prev_action in prev_state
                done (bool): Indicator that the episode is done
            Returns:
                action (int): the action the agent is taking given current_state
            """
            return 1
    env = FloorIsLava(map_name="6x6", slip_rate=0.)
    agent_info = {
            "num_actions": 4,
            "num_states": 36,
            "epsilon": 0.,
            "step_size": 0.1,
            "discount": 0.99,
            "seed": 0
            }
    dummy_agent = DummyAgent()
    dummy_agent.agent_init(agent_info)

    s_true, a_true, r_true = [0, 0, 6, 12, 18, 24, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -100]
    s, a, r = train_episode(dummy_agent, env)
    np.testing.assert_allclose(len(s), len(s_true))
    np.testing.assert_allclose(len(a), len(a_true))
    np.testing.assert_allclose(len(r), len(r_true))
