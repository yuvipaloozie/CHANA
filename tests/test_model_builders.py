from chana.models import builders


def test_checkpoint_builder_uses_training_encoder_initialization(monkeypatch):
    observed = []

    def fake_build_model(architecture, **kwargs):
        observed.append((architecture, kwargs))
        return object()

    monkeypatch.setattr(builders, "build_model", fake_build_model)

    builders.build_checkpoint_model("transunet")
    builders.build_checkpoint_model("unet")
    builders.build_checkpoint_model("unetpp")

    assert observed == [
        ("transunet", {"encoder_weights": "imagenet"}),
        ("unet", {"encoder_weights": None}),
        ("unetpp", {"encoder_weights": None}),
    ]
