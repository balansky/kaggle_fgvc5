import tensorflow as tf
from tensorflow.python.ops import control_flow_ops


#thumbnail decorator
def thumbnail(thumb_size):
    def thumbnail_decorator(decoded_fn):
        def _wraper(encoded_image):
            image = decoded_fn(encoded_image)
            image = tf.expand_dims(image, axis=0)
            max_side = tf.reduce_max(tf.shape(image))
            target_size = tf.constant(thumb_size)
            target_smaller_than_max = tf.less_equal(target_size, max_side)
            resize_size = control_flow_ops.cond(target_smaller_than_max, lambda: max_side, lambda: target_size)
            image = tf.image.resize_image_with_crop_or_pad(image, resize_size, resize_size)
            image = tf.image.resize_bicubic(
                image, [target_size, target_size], align_corners=False)
            image = tf.squeeze(image, squeeze_dims=[0])
            return image

        return _wraper
    return thumbnail_decorator

#standardization decorator
def standardization(decoded_fn):
    def standardization_decorator(encoded_image):
        image = decoded_fn(encoded_image)
        image = tf.image.per_image_standardization(image)
        return image
    return standardization_decorator

#bbox decorator
def bbox(offset_height, offset_width, target_height, target_width):
    def bbox_decorator(decoded_fn):
        def _wraper(encoded_image):
            image = decoded_fn(encoded_image)
            image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)
            return image
        return _wraper
    return bbox_decorator

#bicubic decorator
def bicubic(target_size):
    def bicubic_decorator(decoded_fn):
        def _wraper(encoded_image):
            image = decoded_fn(encoded_image)
            image = tf.expand_dims(image, axis=0)
            image = tf.image.resize_bicubic(
                image, [target_size, target_size], align_corners=False)
            image = tf.squeeze(image, squeeze_dims=[0])
            return image
        return _wraper
    return bicubic_decorator


def distort_bbox(target_bbox, min_object_covered=0.1, aspect_ratio_range=(0.75, 1.33),area_range=(0.05, 1.0),
                max_attempts=100):
    def distort_bbox_decorator(decoded_fn):
        def _wraper(encoded_image):
            decoded_image = decoded_fn(encoded_image)
            crop_bbox = tf.expand_dims(target_bbox, axis=0)
            crop_bbox = tf.expand_dims(crop_bbox, axis=0)
            sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
                tf.shape(decoded_image),
                bounding_boxes=crop_bbox,
                min_object_covered=min_object_covered,
                aspect_ratio_range=aspect_ratio_range,
                area_range=area_range,
                max_attempts=max_attempts,
                use_image_if_no_bounding_boxes=True)
            bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

            # Crop the image to the specified bounding box.
            cropped_image = tf.slice(decoded_image, bbox_begin, bbox_size)
            cropped_image.set_shape([None, None, 3])
            return cropped_image
        return _wraper
    return distort_bbox_decorator

def random_crop(target_size):
    def crop_decorator(decoded_fn):
        def _wraper(encoded_image):
            image = decoded_fn(encoded_image)
            image = tf.random_crop(image, [target_size, target_size, 3])
            return image

        return _wraper

    return crop_decorator



class Decoder(object):

    def __init__(self, output_size, image_resize='default', image_norm='default', distort_color=False, random_flip=False, random_rotate=False):
        self._output_size = output_size
        self._image_norm = image_norm
        self._distort_color = distort_color
        self._random_flip = random_flip
        self._random_rotate = random_rotate
        self.normalize = self.norm_fn(image_norm)
        self.resize = self.resize_fn(image_resize)
        self.resize_method = image_resize


    def minus_one_to_pos_one(self, image):
        image = self.default_norm(image)
        norm_image = tf.subtract(image, 0.5)
        norm_image = tf.multiply(norm_image, 2.0)
        return norm_image

    def default_norm(self, image):
        image = tf.divide(image, 255.)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image

    def standardize_image(self, image):
        image = tf.image.per_image_standardization(image)
        return image

    def no_change_norm(self, image):
        return image

    def decode_image(self, encoded_image):
        image = tf.image.decode_jpeg(encoded_image, channels=3)
        # image = tf.image.decode_image(encoded_image, channels=3)
        return image

    def apply_with_random_selector(self, x, func, num_cases):
        """Computes func(x, sel), with sel sampled from [0...num_cases-1].
        Args:
          x: input Tensor.
          func: Python function to apply.
          num_cases: Python int32, number of cases to sample sel from.
        Returns:
          The result of func(x, sel), where func receives the value of the
          selector as a python integer, but sel is sampled dynamically.
        """
        sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
        # Pass the real x only to one of the func calls.
        return control_flow_ops.merge([func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
                                       for case in range(num_cases)])[0]

    def distort_color(self, image, color_ordering=0, fast_mode=True, scope=None):
        """Distort the color of a Tensor image.
        Each color distortion is non-commutative and thus ordering of the color ops
        matters. Ideally we would randomly permute the ordering of the color ops.
        Rather then adding that level of complication, we select a distinct ordering
        of color ops for each preprocessing thread.
        Args:
          image: 3-D Tensor containing single image in [0, 1].
          color_ordering: Python int, a type of distortion (valid values: 0-3).
          fast_mode: Avoids slower ops (random_hue and random_contrast)
          scope: Optional scope for name_scope.
        Returns:
          3-D Tensor color-distorted image on range [0, 1]
        Raises:
          ValueError: if color_ordering not in [0, 3]
        """
        with tf.name_scope(scope, 'distort_color', [image]):
            if fast_mode:
                if color_ordering == 0:
                    image = tf.image.random_brightness(image, max_delta=32. / 255.)
                    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                else:
                    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                    image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                if color_ordering == 0:
                    image = tf.image.random_brightness(image, max_delta=32. / 255.)
                    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                    image = tf.image.random_hue(image, max_delta=0.2)
                    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                elif color_ordering == 1:
                    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                    image = tf.image.random_brightness(image, max_delta=32. / 255.)
                    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                    image = tf.image.random_hue(image, max_delta=0.2)
                elif color_ordering == 2:
                    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                    image = tf.image.random_hue(image, max_delta=0.2)
                    image = tf.image.random_brightness(image, max_delta=32. / 255.)
                    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                elif color_ordering == 3:
                    image = tf.image.random_hue(image, max_delta=0.2)
                    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                    image = tf.image.random_brightness(image, max_delta=32. / 255.)
                else:
                    raise ValueError('color_ordering must be in [0, 3]')

            # The random_* ops do not necessarily clamp.
            return tf.clip_by_value(image, 0.0, 1.0)

    def decode_and_thumbnail(self, encoded_image, *args, **kwargs):
        @thumbnail(self._output_size)
        def _decode(encoded_image):
            return self.decode_image(encoded_image)
        decoded_image = _decode(encoded_image)
        return decoded_image

    def decode_and_random_crop(self, encoded_image, *args, **kwargs):
        enlarged_size = self._output_size + int(self._output_size * 0.3)
        @random_crop(self._output_size)
        # @thumbnail(enlarged_size)
        @bicubic(enlarged_size)
        def _decode(encoded_image):
            return self.decode_image(encoded_image)
        decoded_image = _decode(encoded_image)
        return decoded_image


    def decode_and_bbox_crop(self, encoded_image, target_bbox, *args, **kwargs):
        target_bbox = tf.cast(target_bbox, tf.int32)
        @bicubic(self._output_size)
        @bbox(target_bbox[0], target_bbox[1], target_bbox[2], target_bbox[3])
        def _decode(encoded_image):
            return self.decode_image(encoded_image)
        decoded_image = _decode(encoded_image)
        return decoded_image


    def decode_and_bicubic_resize(self, encoded_image, *args, **kwargs):
        @bicubic(self._output_size)
        def _decode(encoded_image):
            return self.decode_image(encoded_image)
        decoded_image = _decode(encoded_image)
        return decoded_image

    def decode_and_random_bbox_crop(self, encoded_image, norm_bbox, min_object_covered=0.1,
                                    aspect_ratio_range=(0.75, 1.33),area_range=(0.05, 1.0), max_attempts=100,
                                    *args, **kwargs):
        @bicubic(self._output_size)
        @distort_bbox(norm_bbox, min_object_covered, aspect_ratio_range, area_range, max_attempts)
        def _decode(encoded_image):
            return self.decode_image(encoded_image)
        return _decode(encoded_image)


    def decode(self, encoded_image, *args, **kwargs):
        resized_image = self.resize(encoded_image, *args, **kwargs)
        if self._random_rotate:
            random_degree = tf.random_normal([], 0, 0.25, dtype=tf.float32)
            resized_image = tf.contrib.image.rotate(resized_image, random_degree)
        if self._distort_color:
            normalized_image = self.default_norm(resized_image)
            normalized_image = self.apply_with_random_selector(
                normalized_image,
                lambda x, ordering: self.distort_color(x, ordering, False),
                num_cases=4)
            if self._image_norm == 'minus_one_to_pos_one':
                normalized_image = tf.subtract(normalized_image, 0.5)
                normalized_image = tf.multiply(normalized_image, 2.0)
            elif self._image_norm == 'standardize':
                normalized_image = self.normalize(normalized_image)
        else:
            normalized_image = self.normalize(resized_image)
        if self._random_flip:
            normalized_image = tf.image.random_flip_left_right(normalized_image)
        return normalized_image


    def resize_fn(self, fn_name):
        resize_methods = {
            'thumbnail': self.decode_and_thumbnail,
            'random_crop': self.decode_and_random_crop,
            'bbox_crop': self.decode_and_bbox_crop,
            'bicubic': self.decode_and_bicubic_resize,
            'bbox_random_crop': self.decode_and_random_bbox_crop,
            'default': self.decode_and_bicubic_resize
        }
        return resize_methods[fn_name]


    def norm_fn(self, norm_name):
        normalize_methods = {
            'minus_one_to_pos_one': self.minus_one_to_pos_one,
            'standardize': self.standardize_image,
            'default': self.default_norm,
            'no_change': self.no_change_norm
        }
        return normalize_methods[norm_name]