
import datetime
import re
import os
import csv
import json
from tensorboardX import SummaryWriter

class Tracker:

    def __init__(self, args):

        # Make signature of experiment
        time_signature = str(datetime.datetime.now())[:19]
        time_signature = re.sub('[^0-9]', '_', time_signature)
        signature = '{}_{}_{}'.format(time_signature, args.model_name,
                                      args.dataset_name)

        # Set directory to store run
        self.dir = './runs/{}/'.format(signature)

        