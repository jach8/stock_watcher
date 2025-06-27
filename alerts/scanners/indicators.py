import pandas as pd 
import numpy as np 
import sqlite3 as sql
import json
from dataclasses import dataclass, field
from typing import Optional, Any, Dict
from pathlib import Path
import sys 


@dataclass(slots=True)
class 