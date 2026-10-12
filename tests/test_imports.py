"""Basic import smoke tests for Z-DynGNN."""

def test_imports():
    import zdyngnn
    import zdyngnn.data
    import zdyngnn.dynamics
    import zdyngnn.models
    import zdyngnn.conformal
    import zdyngnn.eval
    assert zdyngnn.__version__ == "0.1.0"

def test_zdyn_gnn_forward():
    import torch
    from zdyngnn.models import ZDynGNN

    model = ZDynGNN(node_in_dim=20, hidden_dim=64)
    x = torch.randn(10, 20)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    mu, sigma = model(x, edge_index)

    assert mu.shape == (10, 1)
    assert sigma.shape == (10, 1)
    assert (sigma > 0).all()
