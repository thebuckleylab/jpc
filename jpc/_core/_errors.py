_PARAM_TYPES = ["sp", "mupc", "ntk"]


def _check_param_type(param_type):
    if param_type not in _PARAM_TYPES:
        raise ValueError(
            'Invalid parameterisation. Options are `"sp"` (standard '
            'parameterisation), `"mupc"` (μPC), or `"ntp"` (neural tangent '
            'parameterisation). See `_get_param_scalings()` (https://thebuckleylab.github.io/jpc/api/Energy%20functions/#jpc._get_param_scalings) '
            'for the specific scalings of these different parameterisations.'
        )
