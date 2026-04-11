import sys
import os
import logging

logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger()