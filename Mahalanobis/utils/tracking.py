
import datetime
import re
import os
import csv
import json
from tensorboardX import SummaryWriter

class Tracker:

    def __init__(self, args):

        # Make signature of experiment
        time_signature = str(datetime.datetime.no