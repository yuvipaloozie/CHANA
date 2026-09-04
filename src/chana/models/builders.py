"""Model definitions extracted from the historical Colab training scripts."""

from __future__ import annotations


def _tf():
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install CHANA with the 'tensorflow' extra to build models") from exc
    return tf


def _layer_output(model, names):
    for name in names:
        try:
            return model.get_layer(name).output
        except ValueError:
            continue
    raise ValueError(f"none of the expected layer names were found: {names}")


def build_unet(input_shape=(512, 512, 3), num_classes=1, encoder_weights=None):
    """DenseNet121-encoder U-Net used in CHANA."""
    tf = _tf()
    layers = tf.keras.layers
    inputs = layers.Input(shape=input_shape, name="input_layer")
    encoder = tf.keras.applications.DenseNet121(
        include_top=False, weights=encoder_weights, input_tensor=inputs
    )
    s1 = _layer_output(encoder, ["conv1/relu", "conv1_relu"])
    s2 = _layer_output(encoder, ["conv2_block6_concat", "conv2/block6/concat"])
    s3 = _layer_output(encoder, ["conv3_block12_concat", "conv3/block12/concat"])
    s4 = _layer_output(encoder, ["conv4_block24_concat", "conv4/block24/concat"])
    bridge = _layer_output(encoder, ["conv5_block16_concat", "conv5/block16/concat"])

    def decoder_block(tensor, skip, filters):
        x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(tensor)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        return layers.Conv2D(filters, 3, padding="same", activation="relu")(x)

    d1 = decoder_block(bridge, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    d5 = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(d4)
    d5 = layers.Conv2D(32, 3, padding="same", activation="relu")(d5)
    output = layers.Conv2D(
        num_classes, 1, activation="sigmoid", dtype="float32", name="final_output"
    )(d5)
    return tf.keras.Model(inputs=inputs, outputs=output, name="Standard_DenseNet_UNet")


def build_unetpp(input_shape=(512, 512, 3), num_classes=1, encoder_weights=None):
    """ResNet50 U-Net++ with four deep-supervision outputs."""
    try:
        from keras_unet_collection import models
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("keras-unet-collection is required for U-Net++") from exc
    return models.unet_plus_2d(
        input_size=input_shape,
        filter_num=[64, 128, 256, 512],
        n_labels=num_classes,
        stack_num_down=4,
        stack_num_up=4,
        activation="ReLU",
        output_activation="Sigmoid",
        batch_norm=True,
        pool=False,
        unpool=False,
        backbone="ResNet50",
        weights=encoder_weights,
        freeze_backbone=False,
        freeze_batch_norm=False,
        deep_supervision=True,
        name="UNetPlusPlus_ResNet50",
    )


def build_transunet(input_shape=(512, 512, 3), num_classes=1, encoder_weights=None):
    """EfficientNetB0 encoder with the study's transformer bottleneck."""
    tf = _tf()
    layers = tf.keras.layers

    class AddPositionEmbedding(layers.Layer):
        def __init__(self, transformer_dim, max_len=65536, **kwargs):
            super().__init__(**kwargs)
            self.pos_embedding = layers.Embedding(max_len, transformer_dim)

        def call(self, x):
            positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
            return x + self.pos_embedding(positions)

    def transformer_bottleneck(tensor, transformer_dim=384, num_heads=6, num_layers=2):
        x = layers.Conv2D(transformer_dim, 1, padding="same", kernel_initializer="he_normal")(tensor)
        x = layers.GroupNormalization(groups=8)(x)
        height, width = int(tensor.shape[1]), int(tensor.shape[2])
        x = layers.Reshape((height * width, transformer_dim))(x)
        x = AddPositionEmbedding(transformer_dim)(x)
        for _ in range(num_layers):
            attention = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=transformer_dim // num_heads, dropout=0.1
            )(x, x)
            x = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, attention]))
            feed_forward = layers.Dense(transformer_dim * 4, activation="relu")(x)
            feed_forward = layers.Dense(transformer_dim)(feed_forward)
            x = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, feed_forward]))
        x = layers.Reshape((height, width, transformer_dim))(x)
        x = layers.Conv2D(512, 1, padding="same", activation="relu", kernel_initializer="he_normal")(x)
        return layers.GroupNormalization(groups=8)(x)

    def decoder_block(tensor, skip, filters):
        x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(tensor)
        x = layers.Lambda(
            lambda values: tf.image.resize(values[0], tf.shape(values[1])[1:3])
        )([x, skip])
        x = layers.Concatenate()([x, skip])
        for _ in range(2):
            x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(x)
            x = layers.GroupNormalization(groups=8)(x)
            x = layers.Activation("relu")(x)
        return x

    inputs = layers.Input(shape=input_shape)
    backbone = tf.keras.applications.EfficientNetB0(
        weights=encoder_weights, include_top=False, input_tensor=inputs
    )
    backbone.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    s1 = backbone.get_layer("block2a_expand_activation").output
    s2 = backbone.get_layer("block3a_expand_activation").output
    s3 = backbone.get_layer("block4a_expand_activation").output
    s4 = backbone.get_layer("block6a_expand_activation").output
    bridge = backbone.get_layer("top_activation").output
    d1 = decoder_block(transformer_bottleneck(bridge), s4, 512)
    d2 = decoder_block(d1, s3, 256)
    aux1 = layers.Conv2D(num_classes, 1, activation="sigmoid", dtype="float32", name="aux1")(d2)
    d3 = decoder_block(d2, s2, 128)
    aux2 = layers.Conv2D(num_classes, 1, activation="sigmoid", dtype="float32", name="aux2")(d3)
    d4 = decoder_block(d3, s1, 64)
    d5 = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(d4)
    d5 = layers.Conv2D(32, 3, padding="same", activation="relu")(d5)
    final = layers.Conv2D(num_classes, 1, activation="sigmoid", dtype="float32", name="final")(d5)
    return tf.keras.Model(inputs=inputs, outputs=[final, aux1, aux2], name="CHANA_TransUNet")


def build_model(architecture: str, **kwargs):
    key = architecture.lower().replace("+", "p").replace("-", "")
    builders = {"unet": build_unet, "unetpp": build_unetpp, "transunet": build_transunet}
    if key not in builders:
        raise ValueError(f"unsupported architecture: {architecture}")
    return builders[key](**kwargs)


def build_checkpoint_model(architecture: str, **kwargs):
    """Build the architecture exactly as initialized before checkpoint loading."""
    key = architecture.lower().replace("+", "p").replace("-", "")
    kwargs.setdefault("encoder_weights", "imagenet" if key == "transunet" else None)
    return build_model(architecture, **kwargs)
