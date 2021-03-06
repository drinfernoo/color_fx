# Color FX
A Home Assistant integration to apply various color effects to smart lights.

### Installation
---
Simply add this repository in HACS, and install it :ok_hand:

To enable the integration, you'll need to add the following to your `configuration.yaml`:
```yaml
color_fx:
  host: !secret internal_url  # your HA instance's local address, only **required** if using color_fx.turn_light_to_matched_color with 'media_player'
```

### Features
---
 * Turn light(s) to recognized colors from a supplied image URL
 * Turn light(s) to random colors
 * Cycle light(s) through colors
 * Automatically handles any resolution of image
 * Flexible options for coloring multiple lights in a single call
 * Flexible options for control over colors

See [Services](https://github.com/drinfernoo/color_fx/wiki/Services) on the wiki for guidance on usage.
