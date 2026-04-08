from resram_core import load_input, resram_data, param_init, raman_residual, run_save
from tqdm import tqdm
import time
from datetime import datetime
import numpy as np
import lmfit

output = resram_data("example")
print(output.abs)
output.plot()
output.fig_raman.savefig("example/example_ramanpng")
output.fig_absfl.savefig("example/example_absfl.png")
output.fig_profs.savefig("example/example_rep.png")