import os
import sys
import numpy as np
import pyroomacoustics as pra
import librosa
import RIR_cls


myObject = RIR_cls.roomIR(datadir='./data', outdir='./outdata')

myObject.simulate_room_IR()