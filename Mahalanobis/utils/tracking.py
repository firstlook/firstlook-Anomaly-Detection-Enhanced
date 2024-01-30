
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

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        # Store settings
        settings_dict = vars(args)

        with open(self.dir + 'settings.json', 'w') as file:
            json.dump(settings_dict, file, sort_keys=True, indent=4)

        # Create csv file for appending stuff during tra