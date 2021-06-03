# -*- coding: utf-8 -*-
import random


def create_random_port():
    rand = random.randrange(50000, 65353)
    dist_url = 'tcp://127.0.0.1:{}'.format(rand)
    return dist_url
