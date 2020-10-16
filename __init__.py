"""Provides an integration that can match the color of lights to the
   'entity_picture' of any supported 'media_player' device."""

import io
import logging
import math

import urllib.request

_LOGGER = logging.getLogger(__name__)

DOMAIN = 'color_fx'
CONF_HOST = 'host'

ATTR_URL = 'url'
ATTR_MEDIA_PLAYER = 'media_player'
ATTR_MODE = 'mode'
ATTR_SAME_COLOR = 'same_color'
ATTR_STEPS = 'steps'
ATTR_START = 'start'

ATTR_ENTITY_PICTURE = 'entity_picture'

GROUP_EXCLUSIVE_IMAGE = 'image'
GROUP_EXCLUSIVE_COLOR_MODE = 'color_mode'
GROUP_EXCLUSIVE_BRIGHTNESS = 'brightness'

MODE_RECOGNIZED = 'recognized'
MODE_COMPLEMENTARY = 'complementary'

ATTR_HUE = 'hue'
ATTR_SATURATION = 'saturation'
ATTR_RED = 'red'
ATTR_GREEN = 'green'
ATTR_BLUE = 'blue'

SERVICE_TURN_LIGHT_TO_MATCHED_COLOR = 'turn_light_to_matched_color'
SERVICE_TURN_LIGHT_TO_RANDOM_COLOR = 'turn_light_to_random_color'
SERVICE_TURN_LIGHT_TO_CYCLE_COLOR = 'turn_light_to_cycle_color'

DEFAULT_IMAGE_RESIZE = (100, 100)
DEFAULT_COLOR = [230, 230, 230]
DEFAULT_SATURATION = [80, 100]
DEFAULT_SATURATION_RANGE = [0, 100]
DEFAULT_BRIGHTNESS = [192, 192]
DEFAULT_BRIGHTNESS_RANGE = [0, 255]
DEFAULT_BRIGHTNESS_PCT = [75, 75]
DEFAULT_BRIGHTNESS_PCT_RANGE = [0, 100]
DEFAULT_BRIGHTNESS_GRAY = 128
DEFAULT_HS_COLORS = {ATTR_HUE: [0, 360], ATTR_SATURATION: [80, 100]}
DEFAULT_RGB_COLORS = {ATTR_RED: [0, 255], ATTR_GREEN: [0, 255], ATTR_BLUE: [0, 255]}
DEFAULT_STEPS = 12
DEFAULT_START = 0
                  
ERROR_MISSING_SOURCE = 'Must have either \'{}\' or \'{}\'.'.format(ATTR_MEDIA_PLAYER, ATTR_URL)
ERROR_INVALID_MODE = 'Must be either \'{}\' or \'{}\'.'

if __name__ != "__main__":
    import voluptuous as vol

    from homeassistant.helpers import config_validation as cv
    from homeassistant.const import (ATTR_ENTITY_ID, SERVICE_TURN_ON)
    from homeassistant.components.light import (ATTR_RGB_COLOR, ATTR_HS_COLOR,
                                                ATTR_BRIGHTNESS, ATTR_BRIGHTNESS_PCT)
    from homeassistant.components import light
    
    CONFIG_SCHEMA = vol.Schema({
        vol.Optional(CONF_HOST): cv.url
    }, extra=vol.ALLOW_EXTRA)
    
    HS_COLOR_SCHEMA = vol.Schema({
        vol.Optional(ATTR_HUE, default=DEFAULT_HS_COLORS[ATTR_HUE]): vol.Any(cv.positive_int, [cv.positive_int]),
        vol.Optional(ATTR_SATURATION, default=DEFAULT_HS_COLORS[ATTR_SATURATION]): vol.Any(cv.positive_int, [cv.positive_int])
    })
    
    RGB_COLOR_SCHEMA = vol.Schema({
        vol.Optional(ATTR_RED, default=DEFAULT_RGB_COLORS[ATTR_RED]): vol.Any(cv.positive_int, [cv.positive_int]),
        vol.Optional(ATTR_GREEN, default=DEFAULT_RGB_COLORS[ATTR_GREEN]): vol.Any(cv.positive_int, [cv.positive_int]),
        vol.Optional(ATTR_BLUE, default=DEFAULT_RGB_COLORS[ATTR_BLUE]): vol.Any(cv.positive_int, [cv.positive_int])
    })
    
    MATCHED_COLOR_SCHEMA = vol.All(
        vol.Schema({
            vol.Required(ATTR_ENTITY_ID): cv.entity_ids,
            vol.Optional(ATTR_SAME_COLOR, default=False): cv.boolean,
            vol.Optional(CONF_HOST): cv.url,
            vol.Exclusive(ATTR_MEDIA_PLAYER, GROUP_EXCLUSIVE_IMAGE): cv.entity_id,
            vol.Exclusive(ATTR_URL, GROUP_EXCLUSIVE_IMAGE): cv.url,
            vol.Optional(ATTR_MODE, default=MODE_RECOGNIZED): vol.Any(MODE_RECOGNIZED, MODE_COMPLEMENTARY,
                msg=ERROR_INVALID_MODE.format(MODE_RECOGNIZED, MODE_COMPLEMENTARY)),
            vol.Exclusive(vol.Optional(ATTR_BRIGHTNESS, default=DEFAULT_BRIGHTNESS[0]),
                GROUP_EXCLUSIVE_BRIGHTNESS): cv.positive_int,
            vol.Exclusive(vol.Optional(ATTR_BRIGHTNESS_PCT, default=DEFAULT_BRIGHTNESS_PCT[0]),
                GROUP_EXCLUSIVE_BRIGHTNESS): cv.positive_int
        }),
        cv.has_at_least_one_key(ATTR_MEDIA_PLAYER, ATTR_URL),
        cv.has_at_most_one_key(ATTR_URL, CONF_HOST)
    )

    RANDOM_COLOR_SCHEMA = vol.All(
        vol.Schema({
            vol.Required(ATTR_ENTITY_ID): cv.entity_ids,
            vol.Optional(ATTR_SAME_COLOR, default=False): cv.boolean,
            vol.Exclusive(vol.Optional(ATTR_HS_COLOR, default=DEFAULT_HS_COLORS),
                GROUP_EXCLUSIVE_COLOR_MODE): HS_COLOR_SCHEMA,
            vol.Exclusive(vol.Optional(ATTR_RGB_COLOR, default=DEFAULT_RGB_COLORS),
                GROUP_EXCLUSIVE_COLOR_MODE): RGB_COLOR_SCHEMA,
            vol.Exclusive(vol.Optional(ATTR_BRIGHTNESS, default=DEFAULT_BRIGHTNESS),
                GROUP_EXCLUSIVE_BRIGHTNESS): vol.Any(cv.positive_int, [cv.positive_int]),
            vol.Exclusive(vol.Optional(ATTR_BRIGHTNESS_PCT, default=DEFAULT_BRIGHTNESS_PCT),
                GROUP_EXCLUSIVE_BRIGHTNESS): vol.Any(cv.positive_int, [cv.positive_int])
        })
    )
    
    CYCLE_COLOR_SCHEMA = vol.All(
        vol.Schema({
            vol.Required(ATTR_ENTITY_ID): cv.entity_ids,
            vol.Optional(ATTR_SAME_COLOR, default=False): cv.boolean,
            vol.Optional(ATTR_STEPS, default=DEFAULT_STEPS): cv.positive_int,
            vol.Optional(ATTR_START, default=DEFAULT_START): cv.positive_int,
            vol.Optional(ATTR_SATURATION, default=DEFAULT_HS_COLORS[ATTR_SATURATION]): vol.Any(cv.positive_int, [cv.positive_int]),
            vol.Exclusive(vol.Optional(ATTR_BRIGHTNESS, default=DEFAULT_BRIGHTNESS),
                GROUP_EXCLUSIVE_BRIGHTNESS): vol.Any(cv.positive_int, [cv.positive_int]),
            vol.Exclusive(vol.Optional(ATTR_BRIGHTNESS_PCT, default=DEFAULT_BRIGHTNESS_PCT),
                GROUP_EXCLUSIVE_BRIGHTNESS): vol.Any(cv.positive_int, [cv.positive_int])
        })
    )


def setup(hass, config):
    def turn_light_to_matched_color(call):
        call_data = dict(call.data)
        comp_config = dict(config[DOMAIN])
        
        color_fx = ColorFX(hass, comp_config)
        mode = call_data.pop(ATTR_MODE)
        same_color = call_data.pop(ATTR_SAME_COLOR)
        
        if ATTR_URL in call_data:
            url = call_data.pop(ATTR_URL)
        elif ATTR_MEDIA_PLAYER in call_data:
            _LOGGER.info('call_data: ' + str(call_data))
            host = ''
            if CONF_HOST not in call_data:
                if CONF_HOST not in comp_config:
                    raise ValueError('\'media_player\' set, but no \'host\' found.')
                else:
                    host = comp_config[CONF_HOST]
            else:
                host = call_data.pop(CONF_HOST)

            entity = call_data.pop(ATTR_MEDIA_PLAYER)
            state = hass.states.get(entity)
            if state:
                attrs = state.attributes
                if ATTR_ENTITY_PICTURE in attrs and attrs[ATTR_ENTITY_PICTURE]:
                    url = '{}{}'.format(host, attrs[ATTR_ENTITY_PICTURE])
                else:
                    _LOGGER.info('{} has no {} attribute.'.format(entity, ATTR_ENTITY_PICTURE))
                    return
            else:
                _LOGGER.info('{} is unavailable in the system.'.format(entity))
                return
                
        brightness_mode = ATTR_BRIGHTNESS
        brightness = DEFAULT_BRIGHTNESS[0]
        if ATTR_BRIGHTNESS_PCT in call_data:
            brightness_mode = ATTR_BRIGHTNESS_PCT
            brightness = call_data.pop(ATTR_BRIGHTNESS_PCT)
        elif ATTR_BRIGHTNESS in call_data:
            brightness = call_data.pop(ATTR_BRIGHTNESS)

        colors = color_fx.matched_colors(url, mode,
                                         len(call_data[ATTR_ENTITY_ID]) + 1)
        colorfulness = list(colors.keys())

        calls = []
        if same_color:
            color = colors[colorfulness[0]]
            brightness = DEFAULT_BRIGHTNESS_GRAY if color[1:] == color[:-1] else brightness
            new_data = {ATTR_ENTITY_ID: call_data[ATTR_ENTITY_ID]}
            new_data[ATTR_RGB_COLOR] = color
            new_data[brightness_mode] = brightness
            calls.append(new_data)
        else:
            for idx, entity in enumerate(call_data[ATTR_ENTITY_ID]):
                color = colors[colorfulness[idx]]
                if mode == MODE_COMPLEMENTARY:
                    color = [abs(c - 255) for c in color]
                brightness = DEFAULT_BRIGHTNESS_GRAY if color[1:] == color[:-1] else brightness

                new_data = {ATTR_ENTITY_ID: [entity]}
                new_data[ATTR_RGB_COLOR] = color
                new_data[brightness_mode] = brightness
                calls.append(new_data)

        for call in calls:
            _LOGGER.debug('Calling {}'.format(call))
            hass.services.call(light.DOMAIN, SERVICE_TURN_ON, call)

    def turn_light_to_random_color(call):
        call_data = dict(call.data)
        
        color_fx = ColorFX(hass, config[DOMAIN])
        color_mode = ATTR_HS_COLOR
        color_ranges = DEFAULT_HS_COLORS
        color_data = {}
        if ATTR_RGB_COLOR in call_data:
            color_mode = ATTR_RGB_COLOR
            color_ranges = DEFAULT_RGB_COLORS
            color_data = call_data.pop(ATTR_RGB_COLOR)
        elif ATTR_HS_COLOR in call_data:
            color_data = call_data.pop(ATTR_HS_COLOR)
            
        for r in [i for i in color_ranges if i in color_data]:
            v = color_data[r]
            color_ranges[r] = v if isinstance(v, list) else [v]
            
        brightness_mode = ATTR_BRIGHTNESS
        brightness_ranges = {ATTR_BRIGHTNESS: DEFAULT_BRIGHTNESS}
        brightness_data = {}
        if ATTR_BRIGHTNESS_PCT in call_data:
            brightness_mode = ATTR_BRIGHTNESS_PCT
            brightness_ranges = DEFAULT_BRIGHTNESS_PCT
            brightness_data = {ATTR_BRIGHTNESS_PCT: call_data.pop(ATTR_BRIGHTNESS_PCT)}
        elif ATTR_BRIGHTNESS in call_data:
            brightness_data = {ATTR_BRIGHTNESS: call_data.pop(ATTR_BRIGHTNESS)}
            
        for r in [i for i in brightness_ranges if i in brightness_data]:
            v = brightness_data[r]
            brightness_ranges[r] = v if isinstance(v, list) else [v]
            
        same_color = call_data.pop(ATTR_SAME_COLOR)

        calls = []
        if same_color:
            color = color_fx.random_color(color_ranges, color_mode)
            brightness = color_fx.random_brightness(brightness_ranges, brightness_mode)
            
            new_data = {ATTR_ENTITY_ID: call_data[ATTR_ENTITY_ID]}
            new_data[color_mode] = color
            new_data[brightness_mode] = brightness
            calls.append(new_data)
        else:
            for entity in call_data[ATTR_ENTITY_ID]:
                color = color_fx.random_color(color_ranges, color_mode)
                brightness = color_fx.random_brightness(brightness_ranges, brightness_mode)

                new_data = {ATTR_ENTITY_ID: [entity]}
                new_data[color_mode] = color
                new_data[brightness_mode] = brightness
                calls.append(new_data)

        for call in calls:
            _LOGGER.debug('Calling {}'.format(call))
            hass.services.call(light.DOMAIN, SERVICE_TURN_ON, call)
            
    def turn_light_to_cycle_color(call):
        call_data = dict(call.data)
        color_fx = ColorFX(hass, config[DOMAIN])
        
        saturation_ranges = call_data.pop(ATTR_SATURATION)
        
        brightness_mode = ATTR_BRIGHTNESS
        brightness_ranges = {ATTR_BRIGHTNESS: DEFAULT_BRIGHTNESS}
        brightness_data = {}
        if ATTR_BRIGHTNESS_PCT in call_data:
            brightness_mode = ATTR_BRIGHTNESS_PCT
            brightness_ranges = DEFAULT_BRIGHTNESS_PCT
            brightness_data = {ATTR_BRIGHTNESS_PCT: call_data.pop(ATTR_BRIGHTNESS_PCT)}
        elif ATTR_BRIGHTNESS in call_data:
            brightness_data = {ATTR_BRIGHTNESS: call_data.pop(ATTR_BRIGHTNESS)}
            
        for r in [i for i in brightness_ranges if i in brightness_data]:
            v = brightness_data[r]
            brightness_ranges[r] = v if isinstance(v, list) else [v]
            
        same_color = call_data.pop(ATTR_SAME_COLOR)
        steps = call_data.pop(ATTR_STEPS)
        current = DEFAULT_START
        start = call_data.pop(ATTR_START)
        
        calls = []
        if same_color:
            entity = call_data[ATTR_ENTITY_ID]
            state = hass.states.get(entity)
            if state:
                attrs = state.attributes
                if ATTR_HS_COLOR in attrs and attrs[ATTR_HS_COLOR]:
                    current = attrs[ATTR_HS_COLOR][0]
            else:
                _LOGGER.info('{} is unavailable in the system.'.format(entity))
                return
            if start > 0:
                current = start
                
            hue = color_fx.cycle_color(current, steps)
            saturation = color_fx.random_saturation(saturation_ranges)
            brightness = color_fx.random_brightness(brightness_ranges, brightness_mode)
            
            new_data = {ATTR_ENTITY_ID: entity}
            new_data[ATTR_HS_COLOR] = [hue, saturation]
            new_data[brightness_mode] = brightness
            calls.append(new_data)
        else:
            for entity in call_data[ATTR_ENTITY_ID]:
                state = hass.states.get(entity)
                if state:
                    attrs = state.attributes
                    if ATTR_HS_COLOR in attrs and attrs[ATTR_HS_COLOR]:
                        current = attrs[ATTR_HS_COLOR][0]
                else:
                    _LOGGER.info('{} is unavailable in the system.'.format(entity))
                    return
                if start > 0:
                    current = start
                
                hue = color_fx.cycle_color(current, steps)
                saturation = color_fx.random_saturation(saturation_ranges)
                brightness = color_fx.random_brightness(brightness_ranges, brightness_mode)

                new_data = {ATTR_ENTITY_ID: [entity]}
                new_data[ATTR_HS_COLOR] = [hue, saturation]
                new_data[brightness_mode] = brightness
                calls.append(new_data)

        for call in calls:
            _LOGGER.debug('Calling {}'.format(call))
            hass.services.call(light.DOMAIN, SERVICE_TURN_ON, call)

    hass.services.register(DOMAIN, SERVICE_TURN_LIGHT_TO_MATCHED_COLOR,
                           turn_light_to_matched_color, schema=MATCHED_COLOR_SCHEMA)
    hass.services.register(DOMAIN, SERVICE_TURN_LIGHT_TO_RANDOM_COLOR,
                           turn_light_to_random_color, schema=RANDOM_COLOR_SCHEMA)
    hass.services.register(DOMAIN, SERVICE_TURN_LIGHT_TO_CYCLE_COLOR,
                           turn_light_to_cycle_color, schema=CYCLE_COLOR_SCHEMA)

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

    def matched_colors(self, url, mode=MODE_RECOGNIZED, top=2):
        best_colors = SpotifyBackgroundColor(url, resize=True).best_colors(k=top * 2, color_tol=5)
        return best_colors

    def random_color(self, ranges=DEFAULT_HS_COLORS, mode=ATTR_HS_COLOR):
        import random
        ranges = self._fix_ranges(ranges)

        if mode == ATTR_HS_COLOR:
            c = [random.randint(ranges[ATTR_HUE][0], ranges[ATTR_HUE][1]),
                 random.randint(ranges[ATTR_SATURATION][0], ranges[ATTR_SATURATION][1])]
            c[0] = clamp(c[0], DEFAULT_HS_COLORS[ATTR_HUE][0], DEFAULT_HS_COLORS[ATTR_HUE][1])
            c[1] = clamp(c[1], DEFAULT_HS_COLORS[ATTR_SATURATION][0], DEFAULT_HS_COLORS[ATTR_SATURATION][1])
        elif mode == ATTR_RGB_COLOR:
            c = [random.randint(ranges[ATTR_RED][0], ranges[ATTR_RED][1]),
                 random.randint(ranges[ATTR_GREEN][0], ranges[ATTR_GREEN][1]),
                 random.randint(ranges[ATTR_BLUE][0], ranges[ATTR_BLUE][1])]
            c[0] = clamp(c[0], DEFAULT_RGB_COLORS[ATTR_RED][0], DEFAULT_RGB_COLORS[ATTR_RED][1])
            c[1] = clamp(c[1], DEFAULT_RGB_COLORS[ATTR_GREEN][0], DEFAULT_RGB_COLORS[ATTR_GREEN][1])
            c[2] = clamp(c[2], DEFAULT_RGB_COLORS[ATTR_BLUE][0], DEFAULT_RGB_COLORS[ATTR_BLUE][1])

        return c
        
    def random_saturation(self, ranges=DEFAULT_SATURATION, mode=ATTR_SATURATION):
        import random
        
        ranges = self._fix_ranges(ranges, mode) 
        c = random.randint(ranges[mode][0], ranges[mode][1])
        c = clamp(c, DEFAULT_SATURATION_RANGE[0], DEFAULT_SATURATION_RANGE[1])

        return c
        
    def random_brightness(self, ranges=DEFAULT_BRIGHTNESS, mode=ATTR_BRIGHTNESS):
        import random
        
        ranges = self._fix_ranges(ranges, mode) 
        c = random.randint(ranges[mode][0], ranges[mode][1])
        if mode == ATTR_BRIGHTNESS:
            c = clamp(c, DEFAULT_BRIGHTNESS_RANGE[0], DEFAULT_BRIGHTNESS_RANGE[1])
        elif mode == ATTR_BRIGHTNESS_PCT:
            c = clamp(c, DEFAULT_BRIGHTNESS_PCT_RANGE[0], DEFAULT_BRIGHTNESS_PCT_RANGE[1])

        return c
        
    def cycle_color(self, current=DEFAULT_START, steps=DEFAULT_STEPS):
        hue = ((360 // steps) + current) % 360
        return hue
    
    def _fix_ranges(self, ranges, mode=0):
        ranges = ranges if isinstance(ranges, dict) else {mode: ranges}

        for range in ranges:
            if len(ranges[range]) == 1:
                ranges[range] = [ranges[range][0], ranges[range][0]]
            elif len(ranges[range]) > 2:
                ranges[range] = [ranges[range][0], ranges[range][1]]

        if any(ranges[i][0] > ranges[i][1] for i in ranges):
              raise ValueError('Lower bound cannot be greater than higher bound for range.')

        return ranges


if __name__ == "__main__":
    params = {'k': 5, 'color_tol': 5}
    print(params)
    image = download_image("https://f4.bcbits.com/img/a2074947048_10.jpg")
    print(SpotifyBackgroundColor(image).best_colors(**params))
    image = download_image("https://upload.wikimedia.org/wikipedia/en/thumb/2/27/Ghost_of_a_rose.jpg/220px-Ghost_of_a_rose.jpg")
    print(SpotifyBackgroundColor(image, resize=True).best_colors(**params))
