"""Tests for deepcontrail.segmentation.models.factory — build_model factory function."""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Output shape and dtype
# ---------------------------------------------------------------------------


def test_build_model_returns_nn_module(dummy_input):
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.segmentation.models import ModelConfig, build_model

    model = build_model(ModelConfig(encoder_name="resnet18", encoder_weights=None))

    assert isinstance(model, torch.nn.Module)


def test_build_model_unet_output_shape(dummy_input):
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.segmentation.models import ModelConfig, build_model

    model = build_model(
        ModelConfig(architecture="Unet", encoder_name="resnet18", encoder_weights=None)
    )
    out = model(dummy_input)

    assert out.shape == (2, 1, 32, 32), f"expected (B,1,H,W), got {out.shape}"


def test_build_model_unetplusplus_output_shape(dummy_input):
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.segmentation.models import ModelConfig, build_model

    model = build_model(
        ModelConfig(architecture="UnetPlusPlus", encoder_name="resnet18", encoder_weights=None)
    )
    out = model(dummy_input)

    assert out.shape == (2, 1, 32, 32), f"expected (B,1,H,W), got {out.shape}"


def test_build_model_deeplabv3plus_output_shape(dummy_input):
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.segmentation.models import ModelConfig, build_model

    model = build_model(
        ModelConfig(
            architecture="DeepLabV3Plus",
            encoder_name="resnet18",
            encoder_weights=None,
        )
    )
    out = model(dummy_input)

    assert out.shape == (2, 1, 32, 32), f"expected (B,1,H,W), got {out.shape}"


def test_build_model_forward_dtype_float32(dummy_input):
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.segmentation.models import ModelConfig, build_model

    model = build_model(ModelConfig(encoder_name="resnet18", encoder_weights=None))
    out = model(dummy_input)

    assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# Backward pass / trainability
# ---------------------------------------------------------------------------


def test_build_model_backward_computes_gradients(dummy_input):
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.segmentation.models import ModelConfig, build_model

    model = build_model(ModelConfig(encoder_name="resnet18", encoder_weights=None))
    out = model(dummy_input)
    loss = out.sum()
    loss.backward()

    has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grads, "backward pass should produce gradients"


# ---------------------------------------------------------------------------
# Architecture quirk handling
# ---------------------------------------------------------------------------


def test_build_model_decoder_attention_stripped_for_deeplabv3(dummy_input):
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.segmentation.models import ModelConfig, build_model

    model = build_model(
        ModelConfig(
            architecture="DeepLabV3Plus",
            encoder_name="resnet18",
            encoder_weights=None,
            decoder_attention_type="scse",
        )
    )
    out = model(dummy_input)

    assert out.shape == (2, 1, 32, 32), "should silently strip unsupported params"


def test_build_model_invalid_architecture_raises():
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.segmentation.models import ModelConfig, build_model

    with pytest.raises(AttributeError):
        build_model(ModelConfig(architecture="NotARealArchitecture"))


# ---------------------------------------------------------------------------
# Encoder / weight variants
# ---------------------------------------------------------------------------


def test_build_model_timm_encoder(dummy_input):
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.segmentation.models import ModelConfig, build_model

    model = build_model(ModelConfig(encoder_name="tu-resnet18", encoder_weights=None))
    out = model(dummy_input)

    assert out.shape == (2, 1, 32, 32)


def test_build_model_no_pretrained_weights(dummy_input):
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.segmentation.models import ModelConfig, build_model

    model = build_model(ModelConfig(encoder_name="resnet18", encoder_weights=None))
    out = model(dummy_input)

    assert out.shape == (2, 1, 32, 32)


def test_build_model_extra_kwargs_forwarded(dummy_input):
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.segmentation.models import ModelConfig, build_model

    model = build_model(
        ModelConfig(
            encoder_name="resnet18",
            encoder_weights=None,
            extra_kwargs={"encoder_depth": 3, "decoder_channels": (128, 64, 32)},
        )
    )
    out = model(dummy_input)

    assert out.shape[0] == 2, "model should accept extra_kwargs"
    assert out.shape[1] == 1


# ---------------------------------------------------------------------------
# Unet++ with transformer-style encoders (no 1/2-scale feature)
# ---------------------------------------------------------------------------

# A tiny timm encoder whose first feature is at 1/4 scale, like CAFormer.
TRANSFORMER_STYLE_ENCODER = "tu-convnext_atto"


def test_build_model_unetplusplus_transformer_encoder_output_shape(dummy_input):
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.segmentation.models import ModelConfig, build_model

    model = build_model(
        ModelConfig(
            architecture="UnetPlusPlus",
            encoder_name=TRANSFORMER_STYLE_ENCODER,
            encoder_weights=None,
        )
    )
    out = model(dummy_input)

    assert out.shape == (2, 1, 32, 32), f"expected (B,1,H,W), got {out.shape}"


def test_build_model_unetplusplus_transformer_encoder_every_parameter_gets_gradient(
    dummy_input,
):
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.segmentation.models import ModelConfig, build_model

    model = build_model(
        ModelConfig(
            architecture="UnetPlusPlus",
            encoder_name=TRANSFORMER_STYLE_ENCODER,
            encoder_weights=None,
        )
    )
    model(dummy_input).sum().backward()

    unused = [n for n, p in model.named_parameters() if p.grad is None or p.numel() == 0]
    assert not unused, f"every layer, including the 1/2-scale path, must be used: {unused}"


def test_build_model_unetplusplus_transformer_encoder_keeps_decoder_options(dummy_input):
    pytest.importorskip("segmentation_models_pytorch")
    from segmentation_models_pytorch.base.modules import SCSEModule

    from deepcontrail.segmentation.models import ModelConfig, build_model

    model = build_model(
        ModelConfig(
            architecture="UnetPlusPlus",
            encoder_name=TRANSFORMER_STYLE_ENCODER,
            encoder_weights=None,
            decoder_attention_type="scse",
            extra_kwargs={"decoder_channels": (128, 64, 48, 32, 32)},
        )
    )
    out = model(dummy_input)

    assert out.shape == (2, 1, 32, 32)
    assert any(isinstance(m, SCSEModule) for m in model.modules()), "attention type was dropped"
    assert model.segmentation_head[0].in_channels == 32, "decoder_channels were dropped"


def test_build_model_unetplusplus_convolutional_encoder_matches_smp():
    pytest.importorskip("segmentation_models_pytorch")
    import segmentation_models_pytorch as smp

    from deepcontrail.segmentation.models import ModelConfig, build_model

    model = build_model(
        ModelConfig(architecture="UnetPlusPlus", encoder_name="resnet18", encoder_weights=None)
    )
    reference = smp.UnetPlusPlus(encoder_name="resnet18", encoder_weights=None, classes=1)

    assert model.state_dict().keys() == reference.state_dict().keys(), (
        "an encoder with a 1/2-scale feature must get the plain smp Unet++"
    )


def test_build_model_unet_transformer_encoder_matches_smp():
    pytest.importorskip("segmentation_models_pytorch")
    import segmentation_models_pytorch as smp

    from deepcontrail.segmentation.models import ModelConfig, build_model

    model = build_model(
        ModelConfig(
            architecture="Unet", encoder_name=TRANSFORMER_STYLE_ENCODER, encoder_weights=None
        )
    )
    reference = smp.Unet(encoder_name=TRANSFORMER_STYLE_ENCODER, encoder_weights=None, classes=1)

    assert model.state_dict().keys() == reference.state_dict().keys(), (
        "only Unet++ gets the stem; other architectures are plain smp"
    )
