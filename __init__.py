"""Provides an integration that can match the color of lights to the
   'entity_picture' of any supported 'media_player' device."""

from collections import OrderedDict
import io
import logging
import math

import urllib.request

DOMAIN = 'color_fx'
_LOGGER = logging.getLogger(__name__)

ATTR_URL = 'url'
ATTR_MODE = 'mode'
ATTR_TOP = 'top'

SERVICE_TURN_LIGHT_TO_MATCHED_COLOR = 'turn_light_to_matched_color'
SERVICE_TURN_LIGHT_TO_RANDOM_COLOR = 'turn_light_to_random_color'

DEFAULT_IMAGE_RESIZE = (100, 100)
DEFAULT_COLOR = [230, 230, 230]

if __name__ != "__main__":
    import voluptuous as vol

    from homeassistant.helpers import config_validation as cv
    from homeassistant.const import (ATTR_ENTITY_ID, SERVICE_TURN_ON)
    from homeassistant.components.light import (ATTR_RGB_COLOR, ATTR_HS_COLOR, ATTR_BRIGHTNESS)
    from homeassistant.components import light

    MATCHED_COLOR_SCHEMA = vol.Schema({
        vol.Required(ATTR_ENTITY_ID): cv.entity_ids,
        vol.Required(ATTR_URL): cv.url,
        vol.Optional(ATTR_MODE): cv.string,
        vol.Optional(ATTR_TOP): cv.positive_int,
    }, extra=vol.ALLOW_EXTRA)

    RANDOM_COLOR_SCHEMA = vol.Schema({
        vol.Required(ATTR_ENTITY_ID): cv.entity_ids,
        vol.Optional(ATTR_MODE): cv.string
    }, extra=vol.ALLOW_EXTRA)


def setup(hass, config):
    def turn_light_to_matched_color(call):
        call_data = dict(call.data)
        color_fx = ColorFX(hass, config[DOMAIN])
        colors = colors_fx.matched_color(call_data.pop(ATTR_URL),
                                         call_data.pop(ATTR_MODE),
                                         call_data.pop(ATTR_TOP))

        new_data = {ATTR_RGB_COLOR: colors}
        if colors[1:] == colors[:-1]:
            new_data[ATTR_BRIGHTNESS] = 128
        call_data.update(new_data)
        _LOGGER.info('Calling {}'.format(call_data))

        hass.services.call(light.DOMAIN, SERVICE_TURN_ON, call_data)

    def turn_light_to_random_color(call):
        call_data = dict(call.data)
        color_fx = ColorFX(hass, config[DOMAIN])
        mode = call_data.pop(ATTR_MODE)
        
        calls = []
        for entity in call_data[ATTR_ENTITY_ID]:
            colors = color_fx.random_color(mode)

            new_data = {ATTR_ENTITY_ID: [entity]}
            new_data[mode] = colors
            new_data[ATTR_BRIGHTNESS] = 192
            calls.append(new_data)
        
        for call in calls:
            _LOGGER.info('Calling {}'.format(call))
            hass.services.call(light.DOMAIN, SERVICE_TURN_ON, call)

    hass.services.register(DOMAIN, SERVICE_TURN_LIGHT_TO_MATCHED_COLOR,
                           turn_light_to_matched_color, schema=MATCHED_COLOR_SCHEMA)
    hass.services.register(DOMAIN, SERVICE_TURN_LIGHT_TO_RANDOM_COLOR,
                           turn_light_to_random_color, schema=RANDOM_COLOR_SCHEMA)

    return True


def download_image(url):
    return io.BytesIO(urllib.request.urlopen(url).read())


def calculate_size(self, original):
    width, height = original
    ratio = width / height
    factor = math.ceil(width / 1000) * 2
    resized = DEFAULT_IMAGE_RESIZE if ratio == 1 else (int(width // factor), int((width // factor) // ratio))
    _LOGGER.info('({}, {}) -> {}'.format(width, height, resized))

    return resized


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
            resized = calculate_size(image.size)
            img = img.resize(resized, resample=Image.BILINEAR)

        self.img = np.array(img).astype(float)

    def best_color(self, k=8, color_tol=10, idx=0):
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

        max_colorful = list(colors.keys())[idx]

        if max_colorful < color_tol:
            # If not colorful, set to default color
            best_color = DEFAULT_COLOR
        else:
            # Pick the most colorful color
            best_color = list(colors.values())[idx]

        return int(best_color[0]), int(best_color[1]), int(best_color[2])

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

    def matched_color(self, url, mode='recognized', top=0):
        if mode in ['recognized', 'complementary']:
            best_color = SpotifyBackgroundColor(url).best_color(k=4, color_tol=5, idx=top)

            return best_color if mode == 'recognized' else [abs(color - 255) for color in best_color]
        else:
            raise ValueError('Invalid Mode. Only \'recognized\' \
                             and \'complementary\' are supported.')

    def random_color(self, mode='hs_color'):
        if mode in ['hs_color', 'rgb_color']:
            p = (360, 101) if mode == 'hs_color' else (256, 256, 256)
        else:
            raise ValueError('Invalid Mode. Only \'rgb_color\' and \'hs_color\' \
                             are supported.')

        import random
        return [random.randint(0, i) for i in p]


if __name__ == "__main__":
    params = {'k': 5, 'color_tol': 5}
    print(params)
    image = download_image("https://f4.bcbits.com/img/a2074947048_10.jpg")
    print(SpotifyBackgroundColor(image, image_processing_size=DEFAULT_IMAGE_RESIZE).best_color(**params))
    image = download_image("https://upload.wikimedia.org/wikipedia/en/thumb/2/27/Ghost_of_a_rose.jpg/220px-Ghost_of_a_rose.jpg")
    print(SpotifyBackgroundColor(image, image_processing_size=DEFAULT_IMAGE_RESIZE).best_color(**params))
