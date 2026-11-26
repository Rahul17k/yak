import tensorflow as tf


def _zoom_out(image: tf.Tensor, scale: tf.Tensor) -> tf.Tensor:
    image_size = tf.shape(image)[:2]
    reduced_size = tf.maximum(
        tf.cast(tf.round(tf.cast(image_size, tf.float32) * scale), tf.int32),
        16,
    )
    reduced = tf.image.resize(image, reduced_size, method="bilinear")
    height_padding = image_size[0] - reduced_size[0]
    width_padding = image_size[1] - reduced_size[1]
    paddings = tf.stack(
        [
            tf.stack([height_padding // 2, height_padding - height_padding // 2]),
            tf.stack([width_padding // 2, width_padding - width_padding // 2]),
            tf.constant([0, 0], dtype=tf.int32),
        ]
    )
    result = tf.pad(reduced, paddings, mode="REFLECT")
    result.set_shape(image.shape)
    return result


def _downsample(image: tf.Tensor, scale: tf.Tensor) -> tf.Tensor:
    image_size = tf.shape(image)[:2]
    reduced_size = tf.maximum(
        tf.cast(tf.round(tf.cast(image_size, tf.float32) * scale), tf.int32),
        16,
    )
    reduced = tf.image.resize(image, reduced_size, method="area")
    result = tf.image.resize(reduced, image_size, method="bilinear")
    result.set_shape(image.shape)
    return result


def _gaussian_blur(image: tf.Tensor, sigma: tf.Tensor) -> tf.Tensor:
    coordinates = tf.cast(tf.range(-3, 4), tf.float32)
    weights = tf.exp(-(coordinates**2) / (2.0 * sigma**2))
    kernel = weights[:, None] * weights[None, :]
    kernel /= tf.reduce_sum(kernel)
    kernel = tf.tile(kernel[:, :, None, None], [1, 1, 3, 1])
    blurred = tf.nn.depthwise_conv2d(
        image[None, ...],
        kernel,
        strides=[1, 1, 1, 1],
        padding="SAME",
    )[0]
    blurred.set_shape(image.shape)
    return blurred


def apply_blur_distance(
    image: tf.Tensor,
    zoom_scale: float,
    resolution_scale: float,
    blur_sigma: float,
) -> tf.Tensor:
    result = tf.cast(image, tf.float32)
    if zoom_scale < 1:
        result = _zoom_out(result, tf.constant(zoom_scale, tf.float32))
    if resolution_scale < 1:
        result = _downsample(result, tf.constant(resolution_scale, tf.float32))
    if blur_sigma > 0:
        result = _gaussian_blur(result, tf.constant(blur_sigma, tf.float32))
    return tf.clip_by_value(result, 0.0, 255.0)


def random_blur_distance(image: tf.Tensor) -> tf.Tensor:
    image = tf.cast(image, tf.float32)
    image = tf.cond(
        tf.random.uniform([]) < 0.30,
        lambda: _zoom_out(image, tf.random.uniform([], 0.55, 0.90)),
        lambda: image,
    )
    image = tf.cond(
        tf.random.uniform([]) < 0.30,
        lambda: _downsample(image, tf.random.uniform([], 0.30, 0.75)),
        lambda: image,
    )
    image = tf.cond(
        tf.random.uniform([]) < 0.35,
        lambda: _gaussian_blur(image, tf.random.uniform([], 0.4, 1.5)),
        lambda: image,
    )
    return tf.clip_by_value(image, 0.0, 255.0)


def blur_distance_views(image: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    return (
        apply_blur_distance(image, 0.82, 0.70, 0.6),
        apply_blur_distance(image, 0.62, 0.42, 1.2),
    )
