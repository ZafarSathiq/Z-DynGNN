"""Basic import smoke tests for SANDCERT."""

def test_imports():
    import sandcert
    import sandcert.data
    import sandcert.dynamics
    import sandcert.models
    import sandcert.conformal
    import sandcert.eval
    assert sandcert.__version__ == "0.1.0"
