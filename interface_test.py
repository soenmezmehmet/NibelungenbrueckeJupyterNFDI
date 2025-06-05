import random
from nibelungenbruecke.scripts.digital_twin_orchestrator.interface_test_DT import DigitalTwinInterface
import os

#%%

path = 'use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/settings/digital_twin_default_parameters.json'
# path = '../../../use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/settings/digital_twin_default_parameters.json'

dt = DigitalTwinInterface(path)
print(dt)
dt.run_model()

