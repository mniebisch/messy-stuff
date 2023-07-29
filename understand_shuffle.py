import numpy as np
from torchdata.datapipes.iter import IterableWrapper
from torchdata.datapipes.iter import Shuffler
dp = IterableWrapper(range(30))
shuffle_dp = dp.shuffle(buffer_size=10)
output = list(shuffle_dp)
output = np.array(output)
output = output.reshape(3, -1)
print(output)