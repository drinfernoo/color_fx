"""Provides an integration that can match the color of lights to the
   'entity_picture' of any supported 'media_player' device."""

import io
import logging
import math

import urllib.request

DOMAIN = 'color_fx'
_LOGGER = logging.getLogger(__name__)

ATTR_URL = 'url'
ATTR_MODE = 'mode'
ATTR_SAME_COLOR = 'same_color'

ATTR_RANGE_HUE = 'range_hue'
ATTR_RANGE_SAT = 'range_sat'
ATTR_RANGE_RED = 'range_red'
ATTR_RANGE_GREEN = 'range_green'
ATTR_RANGE_BLUE = 'range_blue'

SERVICE_TURN_LIGHT_TO_MATCHED_COLOR = 'turn_light_to_matched_color'
SERVICE_TURN_LIGHT_TO_RANDOM_COLOR = 'turn_light_to_random_color'

DEFAULT_IMAGE_RESIZE = (100, 100)
DEFAULT_COLOR = [230, 230, 230]
DEFAULT_RANGES = {ATTR_RANGE_HUE: (0, 360),
                  ATTR_RANGE_SAT: (80, 100),
                  ATTR_RANGE_RED: (0, 255),
                  ATTR_RANGE_GREEN: (0, 255),
                  ATTR_RANGE_BLUE: (0, 255)}

if __name__ != "__main__":
    import voluptuous as vol

    from homeassistant.helpers import config_validation as cv
    from homeassistant.const import (ATTR_ENTITY_ID, SERVICE_TURN_ON)
    from homeassistant.components.light import (ATTR_RGB_COLOR, ATTR_HS_COLOR, ATTR_BRIGHTNESS)
    from homeassistant.components import light

    MATCHED_COLOR_SCHEMA = vol.Schema({
        vol.Required(ATTR_ENTITY_ID): cv.entity_ids,
        vol.Required(ATTR_URL): cv.url,
        vol.Optional(ATTR_MODE, default='recognized'): cv.string,
        vol.Optional(ATTR_SAME_COLOR, default=False): cv.boolean,
        vol.Optional(ATTR_RANGE_HUE, default=(0, 360)): cv.positive_int,
        vol.Optional(ATTR_RANGE_SAT, default=(80, 100)): cv.positive_int,
        vol.Optional(ATTR_RANGE_RED, default=(0, 255)): cv.positive_int,
        vol.Optional(ATTR_RANGE_GREEN, default=(0, 255)): cv.positive_int,
        vol.Optional(ATTR_RANGE_BLUE, default=(0, 255)): cv.positive_int
    }, extra=vol.ALLOW_EXTRA)

    RANDOM_COLOR_SCHEMA = vol.Schema({
        vol.Required(ATTR_ENTITY_ID): cv.entity_ids,
        vol.Optional(ATTR_MODE, default='hs_color'): cv.string,
        vol.Optional(ATTR_SAME_COLOR, default=False): cv.boolean
    }, extra=vol.ALLOW_EXTRA)


def setup(hass, config):
    def turn_light_to_matched_color(call):
        call_data = dict(call.data)
        color_fx = ColorFX(hass, config[DOMAIN])
        mode = call_data.pop(ATTR_MODE)
        same_color = call_data.pop(ATTR_SAME_COLOR)

        colors = color_fx.matched_colors(call_data.pop(ATTR_URL),
                                         mode,
                                         len(call_data[ATTR_ENTITY_ID]) + 1)
        colorfulness = list(colors.keys())

        calls = []
        if same_color:
            color = colors[colorfulness[0]]
            new_data = {ATTR_ENTITY_ID: call_data[ATTR_ENTITY_ID]}
            new_data[ATTR_RGB_COLOR] = color
            new_data[ATTR_BRIGHTNESS] = 128 if color[1:] == color[:-1] else 192
            calls.append(new_data)
        else:
            for idx, entity in enumerate(call_data[ATTR_ENTITY_ID]):
                color = colors[colorfulness[idx]]
                if mode == 'complementary':
                    color = [abs(c - 255) for c in color]

                new_data = {ATTR_ENTITY_ID: [entity]}
                new_data[ATTR_RGB_COLOR] = color
                new_data[ATTR_BRIGHTNESS] = 128 if color[1:] == color[:-1] else 192
                calls.append(new_data)

        for call in calls:
            _LOGGER.debug('Calling {}'.format(call))
            hass.services.call(light.DOMAIN, SERVICE_TURN_ON, call)

    def turn_light_to_random_color(call):
        call_data = dict(call.data)
        color_fx = ColorFX(hass, config[DOMAIN])
        mode = call_data.pop(ATTR_MODE)
        same_color = call_data.pop(ATTR_SAME_COLOR)

        ranges = DEFAULT_RANGES
        range_keys = list(ranges.keys())
        for idx, r in enumerate(range_keys):
            if r in call_data:
                ranges[r] = list(call_data.pop(r))

        calls = []
        if same_color:
            color = color_fx.random_color(mode, ranges)
            new_data = {ATTR_ENTITY_ID: call_data[ATTR_ENTITY_ID]}
            new_data[mode] = color
            new_data[ATTR_BRIGHTNESS] = 192
            calls.append(new_data)
        else:
            for entity in call_data[ATTR_ENTITY_ID]:
                color = color_fx.random_color(mode, ranges)

                new_data = {ATTR_ENTITY_ID: [entity]}
                new_data[mode] = color
                new_data[ATTR_BRIGHTNESS] = 192
                calls.append(new_data)

        for call in calls:
            _LOGGER.debug('Calling {}'.format(call))
            hass.services.call(light.DOMAIN, SERVICE_TURN_ON, call)

    hass.services.register(DOMAIN, SERVICE_TURN_LIGHT_TO_MATCHED_COLOR,
                           turn_light_to_matched_color, schema=MATCHED_COLOR_SCHEMA)
    hass.services.register(DOMAIN, SERVICE_TURN_LIGHT_TO_RANDOM_COLOR,
                           turn_light_to_random_color, schema=RANDOM_COLOR_SCHEMA)

    return True


def download_image(url):
    return io.BytesIO(urllib.request.urlopen(url).read())


def calculate_size(original):
    width, height = original
    ratio = width / height
    factor = math.ceil(width / 1000) * 2
    resized = DEFAULT_IMAGE_RESIZE if ratio == 1 else (int(width // factor), int((width // factor) // ratio))
    _LOGGER.debug('Resizing from ({}, {}) to {}'.format(width, height, resized))

    return resized


def clamp(n, low, high):
    return max(low, min(n, high))


""" Credit to: https://github.com/davidkrantz/Colorfy for original
    implementation."""
class SpotifyBackgroundColor:
    """Analyzes an image and finds a fitting background color.

    Main use is to analyze album artwork and calculate the background
    color Spotify sets when playing on a Chromecast.

    Attributes:
        img (ndarray): The image to analyze.

    """

    def __init__(self, url, format='RGB', resize=None, crop=False):
        """Prepare the image for analyzation.

        Args:
            img (ndarray): The image to analyze.
            format (str): Format of `img`, either RGB or BGR.
            image_processing_size: (int/float/tuple): Process image or not.
                int - Percentage of current size.
                float - Fraction of current size.
                tuple - Size of the output image (must be integers).

        Raises:
            ValueError: If `format` is not RGB or BGR.

        """
        import numpy as np
        from PIL import Image

        img = Image.open(download_image(url))
        if format in ['RGB', 'BGR']:
            img = img.convert(format)
        else:
            raise ValueError('Invalid format. Only RGB and BGR image ' \
                             'format supported.')

        if crop:
            img = self.crop_center(img, 512, 512)

        if resize:
            resized = calculate_size(img.size)
            img = img.resize(resized, resample=Image.BILINEAR)

        self.img = np.array(img).astype(float)

    def best_colors(self, k=8, color_tol=10):
        """Returns a suitable background color for the given image.

        Uses k-means clustering to find `k` distinct colors in
        the image. A colorfulness index is then calculated for each
        of these colors. The color with the highest colorfulness
        index is returned if it is greater than or equal to the
        colorfulness tolerance `color_tol`. If no color is colorful
        enough, a gray color will be returned. Returns more or less
        the same color as Spotify in 80 % of the cases.

        Args:
            k (int): Number of clusters to form.
            color_tol (float): Tolerance for a colorful color.
                Colorfulness is defined as described by Hasler and
                Süsstrunk (2003) in https://infoscience.epfl.ch/
                record/33994/files/HaslerS03.pdf.
            plot (bool): Plot the original image, k-means result and
                calculated background color. Only used for testing.

        Returns:
            tuple: (R, G, B). The calculated background color.

        """
        from scipy.cluster.vq import kmeans

        self.img = self.img.reshape((self.img.shape[0]*self.img.shape[1], 3))

        centroids = kmeans(self.img, k)[0]

        colorfulness = [self.colorfulness(color[0], color[1], color[2]) for color in centroids]
        paired_colorfulness = {colorfulness[i]: centroids[i] for i in range(len(colorfulness))}
        colors = {c: paired_colorfulness[c] for c in sorted(paired_colorfulness, reverse=True)}

        for c in colors:
            if c < color_tol:
                colors[c] = DEFAULT_COLOR
            else:
                colors[c] = [int(i) for i in colors[c]]

        return colors

    def colorfulness(self, r, g, b):
        """Returns a colorfulness index of given RGB combination.

        Implementation of the colorfulness metric proposed by
        Hasler and Süsstrunk (2003) in https://infoscience.epfl.ch/
        record/33994/files/HaslerS03.pdf.

        Args:
            r (int): Red component.
            g (int): Green component.
            b (int): Blue component.

        Returns:
            float: Colorfulness metric.

        """
        import numpy as np

        rg = np.absolute(r - g)
        yb = np.absolute(0.5 * (r + g) - b)

        # Compute the mean and standard deviation of both `rg` and `yb`.
        rb_mean, rb_std = (np.mean(rg), np.std(rg))
        yb_mean, yb_std = (np.mean(yb), np.std(yb))

        # Combine the mean and standard deviations.
        std_root = np.sqrt((rb_std ** 2) + (yb_std ** 2))
        mean_root = np.sqrt((rb_mean ** 2) + (yb_mean ** 2))

        return std_root + (0.3 * mean_root)

    def crop_center(self, img, cropx, cropy):
        import numpy as np
        from PIL import Image

        img = np.array(img)
        y, x = img.shape[:2]
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return Image.fromarray(img[starty:starty + cropy, startx:startx + cropx])


class ColorFX:
    def __init__(self, hass, component_config):
        self.hass = hass
        self.config = component_config

    def matched_colors(self, url, mode='recognized', top=2):
        if mode in ['recognized', 'complementary']:
            best_colors = SpotifyBackgroundColor(url, resize=True).best_colors(k=top * 2, color_tol=5)

            return best_colors
        else:
            raise ValueError('Invalid Mode. Only \'recognized\' \
                             and \'complementary\' are supported.')

    def random_color(self, mode='hs_color', ranges=None):
        if mode in ['hs_color', 'rgb_color']:
            ranges = ranges if ranges else DEFAULT_RANGES

            for range in ranges:
                if len(ranges[range]) == 1:
                    ranges[range] = [ranges[range][0], ranges[range][0]]
                elif len(ranges[range]) > 2:
                    ranges[range] = [ranges[range][0], ranges[range][1]]

            if any(ranges[i][0] > ranges[i][1] for i in ranges):
                  raise ValueError('Lower bound cannot be greater than higher bound for range.')

            import random
            if mode == 'hs_color':
                c = [random.randint(ranges[ATTR_RANGE_HUE][0], ranges[ATTR_RANGE_HUE][1]),
                     random.randint(ranges[ATTR_RANGE_SAT][0], ranges[ATTR_RANGE_SAT][1])]
                c[0] = clamp(c[0], DEFAULT_RANGES[ATTR_RANGE_HUE][0], DEFAULT_RANGES[ATTR_RANGE_HUE][1])
                c[1] = clamp(c[1], DEFAULT_RANGES[ATTR_RANGE_SAT][0], DEFAULT_RANGES[ATTR_RANGE_SAT][1])
            elif mode == 'rgb_color':
                c = [random.randint(ranges[ATTR_RANGE_RED][0], ranges[ATTR_RANGE_RED][1]),
                     random.randint(ranges[ATTR_RANGE_GREEN][0], ranges[ATTR_RANGE_GREEN][1]),
                     random.randint(ranges[ATTR_RANGE_BLUE][0], ranges[ATTR_RANGE_BLUE][1])]
                c[0] = clamp(c[0], DEFAULT_RANGES[ATTR_RANGE_RED][0], DEFAULT_RANGES[ATTR_RANGE_RED][1])
                c[1] = clamp(c[1], DEFAULT_RANGES[ATTR_RANGE_GREEN][0], DEFAULT_RANGES[ATTR_RANGE_GREEN][1])
                c[2] = clamp(c[2], DEFAULT_RANGES[ATTR_RANGE_BLUE][0], DEFAULT_RANGES[ATTR_RANGE_BLUE][1])
        else:
            raise ValueError('Invalid Mode. Only \'rgb_color\' and \'hs_color\' \
                             are supported.')

        return c


if __name__ == "__main__":
    params = {'k': 5, 'color_tol': 5}
    print(params)
    image = download_image("https://f4.bcbits.com/img/a2074947048_10.jpg")
    print(SpotifyBackgroundColor(image).best_colors(**params))
    image = download_image("https://upload.wikimedia.org/wikipedia/en/thumb/2/27/Ghost_of_a_rose.jpg/220px-Ghost_of_a_rose.jpg")
    print(SpotifyBackgroundColor(image, resize=True).best_colors(**params))
