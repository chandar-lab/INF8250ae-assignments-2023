from otter.test_files import test_case

OK_FORMAT = False

name = "question 3b"
points = 5

@test_case(points=1, hidden=False)
def test_value_init_network_type(gym, np, torch, functools, value_init_network):
    torch.manual_seed(0)
    np.random.seed(0)
    test_env = gym.make('CartPole-v1')
    test_output = value_init_network(test_env)
    input_layer_dims = test_output[0].in_features, test_output[0].out_features
    layer1_dims = test_output[2].in_features, test_output[2].out_features
    layer2_dims = test_output[4].in_features, test_output[4].out_features
    np.testing.assert_allclose(input_layer_dims, (4,32))
    np.testing.assert_allclose(layer1_dims, (32,32))
    np.testing.assert_allclose(layer2_dims, (32,1))
@test_case(points=2, hidden=False)
def test_value_init_network_type_1(gym, np, torch, functools, value_init_network):
    torch.manual_seed(0)
    np.random.seed(0)
    test_env = gym.make('CartPole-v1')
    test_output = value_init_network(test_env)
    test_state = torch.tensor([-0.04, 0.02, -0.04, 0.02])
    test_out = test_output(test_state)
    np.testing.assert_allclose(test_out.detach().cpu().numpy(), np.array([0.113974]), rtol = 1e-5, atol = 0)
