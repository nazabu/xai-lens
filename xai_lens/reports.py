import numpy as np
import pandas as pd
import sklearn.metrics import confusion_martix

class BiasDetector:
    def __init__(self, model, data, target, sensitive_features):