from pathlib import Path
import io
import base64
import argparse
import time
from datetime import datetime, timedelta
from typing import Any, Optional

import pickle
import numpy as np
import numpy.typing as npt
import zmq

from fastmcp import FastMCP
from fastmcp.utilities.types import Image as mcpImage

from fastmcp.resources import FileResource
from pathlib import Path
from fastmcp.utilities.types import Image as mcpImage
from datetime import datetime, timedelta
from typing import Any, Optional

import requests
from pydantic import AnyHttpUrl, BaseModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
from requests.exceptions import HTTPError, RequestException

import h5py
import mfid

import gymnasium as gym
from gymnasium import spaces

import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from Play_UI_v6_c1a1only import PlayUI

class Microscope_Client():
    '''Communicates with the server on the microscope PC.'''
    def __init__(self, host='192.168.0.24', port=7001):
        try:
            # Set timeout in milliseconds
            timeout_ms = 300000  # 5 minutes
            context = zmq.Context()
            self.ClientSocket = context.socket(zmq.REQ)
            self.ClientSocket.setsockopt(zmq.RCVTIMEO, timeout_ms)
            self.ClientSocket.setsockopt(zmq.SNDTIMEO, timeout_ms)
            self.ClientSocket.connect(f"tcp://{host}:{port}")
        except ConnectionRefusedError:
            print('Please start the BEACON server and try again...')
            exit()
    
    def send_traffic(self, message):
        '''
        Sends and receives messages from the server.
        
        Parameters
        ----------
        message : dict
            Message for the server.
        
        Returns
        -------
        : dict or None
            Response from the server. If no repsonse then None.
        '''
        print(f'Microscope_Client: {message}')
        try:
            self.ClientSocket.send(pickle.dumps(message))
            response = pickle.loads(self.ClientSocket.recv())
            return response
            
        except zmq.Again:
            print("Timeout occurred")
            return None

# Microscope (BEACON) server commands
def acquire_ceos_tableau(angle=18, tabType='fast'):
    """ Acquire a tableau. Hard coded to fast with 18 mrad."""
    d = {'type': 'tableau', 'angle': angle, 'tabType': tabType}
    Response = microscope_client.send_traffic(d)
    if Response is not None:
        if Response['reply_data'] is None:
            raise Exception('Command failed.')
        else:
            reply_data = Response['reply_data']
            return reply_data
    else:
        raise Exception('No response received.')

def acquire_c1a1(WD_x=0.0, WD_y=0.0):
    """ Tilt and acquire a C1A1 measurement. WD is in mrad."""
    d = {'type': 'c1a1', 'ab_values':{'WD_x':WD_x, 'WD_y':WD_y}}
    Response = microscope_client.send_traffic(d)
    if Response is not None:
        if Response['reply_data'] is None:
            raise Exception('Command failed.')
        else:
            reply_data = Response['reply_data']
            return reply_data

def change_aberrations(ab_values:dict):
    '''
    Change aberrations relative to the current values by the indicated amount. 
    This is a delta from the current value.
    Aberrations are NOT reset to current values after function call.
    Some common names of the aberrations are:
    C1 is one-dimensional.
    A1 is 2-fold astigmatism and has an x and y component
    B2 is coma and has an x and y component
    C3 is the thrid-order shperical aberration (sometimes just called the spherical aberration) and is one-dimensional.

    Parameters
    ----------
    ab_values : dict
        Dictionary of values by which to change aberrations. Values are in metres. 
        Keys are 'C1', 'A1_x', 'A1_y', 'B2_x', 'B2_y', 'A2_x', 'A2_y', 'C3', 'S3_x', 'S3_y', 'A3_x', 'A3_y'.
        The values for each aberration are a float.

    Returns
    -------
    None.

    '''
    ab_select = {'C1': None,
                 'A1_x': 'coarse',
                 'A1_y': 'coarse',
                 'B2_x': 'coarse',
                 'B2_y': 'coarse',
                 'A2_x': 'coarse',
                 'A2_y': 'coarse',
                 'C3': None,
                 'A3_x': 'coarse',
                 'A3_y': 'coarse',
                 'S3_x': 'coarse',
                 'S3_y': 'coarse',
                 }

    C1_defocus_flag = True
    undo = False
    bscomp = False
    
    d = {'type': 'ab_only',
         'ab_values': ab_values,
         'ab_select': ab_select,
         'C1_defocus_flag': C1_defocus_flag,
         'undo': undo,
         'bscomp': bscomp,
         }
    Response = microscope_client.send_traffic(d)
    if Response is not None:
        return Response
    

class CorrectorExperimentEnv(gym.Env):
    """Gym env with same action_table as CorrectorPlayEnv for C1A1 mode."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    KEYS = ["C1", "A1", "A2", "B2", "A3", "S3", "C3"]
    #MAIN_KEYS = ["A2", "B2", "A3", "S3"]
    A1_KEYS = ["A1"]


    PCT_CHOICES = [10, 20, 30, 50, 100]
    #C3_PCT_CHOICES = [1, 5, 10, 15, 20, 30, 50, 100]
    C1_PCT_CHOICES = [1, 5, 10, 15, 20, 30, 50, 100]

    def __init__(
            self,
            render_mode: str | None = None,
            setting_seed: int = 0,
            max_steps: int = 200,
            explode_keys=None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.explode_keys = explode_keys if explode_keys is not None else list(self.KEYS)

        self.action_table = self._build_action_table()
        self.action_space = spaces.Discrete(len(self.action_table))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.KEYS),), dtype=np.float32)

        self.t = 0
        self.params = {k: 0.0 for k in self.KEYS}

        self.setting_seed = int(setting_seed)
        self.setting_rng = np.random.default_rng(self.setting_seed)
        self.error_rng = np.random.default_rng(self.setting_seed + 1)

    def _build_action_table(self):
        table = []
        # Comment out high-order actions (MAIN_KEYS + A1_KEYS and C3)
        # for p in self.MAIN_KEYS + self.A1_KEYS:
        #     for pct in self.PCT_CHOICES:
        #         table.append({"type": "pct_button", "target": p, "pct": int(pct), "key": (p, int(pct))})

        # for pct in self.C3_PCT_CHOICES:
        #     table.append({"type": "pct_button", "target": "C3", "pct": int(pct), "key": ("C3", int(pct))})

        # Keep only low-order C1 actions
        for pct in self.C1_PCT_CHOICES:
            table.append({"type": "pct_button", "target": "C1", "pct": int(pct), "key": ("C1", int(pct))})

        # And A1 actions (low-order)
        for pct in self.PCT_CHOICES:
            table.append({"type": "pct_button", "target": "A1", "pct": int(pct), "key": ("A1", int(pct))})

        return table

    def _obs(self):
        c1a1_reply = acquire_c1a1(WD_x=0.0, WD_y=0.0)
        self.params['C1'] = c1a1_reply['C1'][0]
        self.params['A1'] = np.sqrt(c1a1_reply['A1'][0]**2 + c1a1_reply['A1'][1]**2)

        obs = np.array([self.params[k] for k in self.KEYS], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.params = {k: 0.0 for k in self.KEYS}
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        act = self.action_table[int(action)]
        target = act["target"]
        pct = act["pct"] / 100.0

        if target == "C1":
            current = acquire_c1a1(WD_x=0.0, WD_y=0.0)["C1"][0]
            ab_changes = {"C1": -pct * current}
            change_aberrations(ab_changes)

        if target == "A1":
            current = acquire_c1a1(WD_x=0.0, WD_y=0.0)["A1"]
            ab_changes = {"A1_x": pct * current[0], "A1_y": pct * current[1]}
            change_aberrations(ab_changes)

        obs = self._obs()
        total_dev = float(sum(abs(v) for v in self.params.values()))
        terminated = bool(all(abs(self.params[k]) < 20.0 for k in self.KEYS))
        deviation_exploded = total_dev > 100000.0
        reward = -1000.0 if deviation_exploded else 0.0
        truncated = (self.t >= self.max_steps or deviation_exploded) and (not terminated)

        info = {"deviation_exploded": deviation_exploded, "goal_achieved": terminated,"action": act}

        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        if self.render_mode != "human":
            return
        print(f"t={self.t} params={self.params}")


def main():
    # Acquire a tableau
    # tableau_reply = acquire_ceos_tableau(angle=25, tabType='standard')
    # print(f'Tableau acquired: {tableau_reply}')



    # Change aberrations by some amount (e.g. change A1_x by 10 nm and A1_y by -5 nm)
    #ab_changes = {'A1_x': 10e-8, 'A1_y': -5e-8}
    #abchanges_reply = change_aberrations(ab_changes)
    #print(f'Aberrations changed: {abchanges_reply}')
    
    # Acquire a C1A1 measurement with 18 mrad WD in both x and y
    # c1a1_reply = acquire_c1a1(WD_x=0.0, WD_y=0.0)
    # print(f'C1A1 acquired: {c1a1_reply}')

    env = CorrectorExperimentEnv(
            render_mode=None,
            setting_seed=0,
            # init_seed=123,
            # error_seed=999,
            max_steps=500,
            # couple_prob_pct=0.5,
            # user_gamma={
            #     'C1-A1':0.0005,
            #     'A1-C1':0.24,
            #     'B2-A1':-0.912/10, 'B2-C1':-1.82/10,
            #     'A2-A1':-1.244/10, 'A2-C1':-0.637/10, 'A2-B2':-1.18/5,
            #     'C3-C1':-0.72/100, 'C3-A1':-0.44/100, 'C3-A2':+0.967/10, 'C3-B2':+0.882/10, 'C3-S3':+0.345,
            #     'S3-A1':-0.325/100,'S3-C1':-1.332/100,'S3-A2':-0.777/20,'S3-B2':-0.577/20,'S3-A3':0.2,'S3-C3':0.23
            #     },
            # user_beta={},
            # user_sigma={'C1':1,'A1':1,'A2':100,'B2':20,'C3':200,'A3':100,'S3':100},
            # init_ranges={'C1': (0, 20), 'A1': (0, 50), 'A2': (100, 500), 'B2': (100, 500),
            #                 'C3': (800, 3000), 'S3': (1000, 3000), 'A3': (1000, 3000)}
    )

    root = tk.Tk()
    app = PlayUI(root, env)
    root.mainloop()


if __name__ == "__main__":
    # Initialize the Microscope client with
    # TEAM 0.5 microscope PC connection settings
    mhost = '192.168.0.24'
    mport = 7001
    microscope_client = Microscope_Client(mhost, mport)

    # Run the main function
    main()