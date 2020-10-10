import logging
import io
import urllib.request

DEFAULT_IMAGE_RESIZE = (100, 100)

if __name__ != "__main__":
    import voluptuous as vol

    from homeassistant.helpers import config_validation as cv
    from homeassistant.const import (ATTR_ENTITY_ID, SERVICE_TURN_ON)
    from homeassistant.components.light import (ATTR_RGB_COLOR)
    from homeassistant.components import light

    _LOGGER = logging.getLogger(__name__)

    ATTR_URL = 'url'
    SERVICE_RECOGNIZE_COLOR_AND_SET_LIGHT = 'turn_light_to_recognized_color'
    DOMAIN = 'color_recognizer'

    RECOGNIZE_COLOR_SCHEMA = vol.Schema({
        vol.Required(ATTR_URL): cv.url,
        vol.Required(ATTR_ENTITY_ID): cv.entity_ids,
    }, extra=vol.ALLOW_EXTRA)


def setup(hass, config):
    def turn_lights_to_recognized_color(call):
        call_data = dict(call.data)
        colors = ColorRecognizer(hass, config[DOMAIN], call_data.pop(ATTR_URL)).best_colors()
        call_data.update({ATTR_RGB_COLOR: colors})

        hass.services.call(light.DOMAIN, SERVICE_TURN_ON, call_data)

    hass.services.register(DOMAIN, SERVICE_RECOGNIZE_COLOR_AND_SET_LIGHT, turn_lights_to_recognized_color, schema=RECOGNIZE_COLOR_SCHEMA)

    return True

def download_image(url):
    return io.BytesIO(urllib.request.urlopen(url).read())


""" Taken from: https://github.com/davidkrantz/Colorfy """
class SpotifyBackgroundColor:
    """Analyzes an image and finds a fitting background color.

    Main use is to analyze album artwork and calculate the background
    color Spotify sets when playing on a Chromecast.

    Attributes:
        img (ndarray): The image to analyze.

    """

    def __init__(self, img, format='RGB', image_processing_size=None):
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

        img = np.array(Image.open(img))

        if format == 'RGB':
            self.img = img
        elif format == 'BGR':
            self.img = self.img[..., ::-1]
        else:
            raise ValueError('Invalid format. Only RGB and BGR image ' \
                             'format supported.')

        if image_processing_size:
            self.img = np.array(Image.fromarray(self.img).resize(image_processing_size, resample=Image.BILINEAR))

        self.img = self.img.astype(float)

    def best_color(self, k=8, color_tol=10):
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
        import numpy as np
        from scipy.cluster.vq import kmeans

        self.img = self.img.reshape((self.img.shape[0]*self.img.shape[1], 3))

        centroids = kmeans(self.img, k)[0]

        colorfulness = [self.colorfulness(color[0], color[1], color[2]) for color in centroids]
        max_colorful = np.max(colorfulness)

        if max_colorful < color_tol:
            # If not colorful, set to gray
            best_color = [230, 230, 230]
        else:
            # Pick the most colorful color
            best_color = centroids[np.argmax(colorfulness)]

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


class ColorRecognizer:
    def __init__(self, hass, component_config, url):
        self.hass = hass
        self.config = component_config
        self.url = url

    def best_colors(self):
        image = download_image(self.url)
        return SpotifyBackgroundColor(image, image_processing_size=DEFAULT_IMAGE_RESIZE).best_color(k=4, color_tol=5)

if __name__ == "__main__":
    params = {'k': 5, 'color_tol': 5}
    print(params)
    image = download_image("https://f4.bcbits.com/img/a2074947048_10.jpg")
    print(SpotifyBackgroundColor(image, image_processing_size=DEFAULT_IMAGE_RESIZE).best_color(**params))
    image = download_image("https://upload.wikimedia.org/wikipedia/en/thumb/2/27/Ghost_of_a_rose.jpg/220px-Ghost_of_a_rose.jpg")
    print(SpotifyBackgroundColor(image, image_processing_size=DEFAULT_IMAGE_RESIZE).best_color(**params))
