from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass

from laminocupy_cli.config import *
from laminocupy_cli.fbp_filter import *
from laminocupy_cli.logging import *
from laminocupy_cli.rec_steps import *
from laminocupy_cli.remove_stripe import *
from laminocupy_cli.retrieve_phase import *
from laminocupy_cli.utils import *
