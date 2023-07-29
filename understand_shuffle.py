from torchdata.datapipes.iter import IterableWrapper
from torchdata.datapipes.iter import Shuffler
from torchdata import dataloader2


dp = IterableWrapper(range(30))
shuffle_dp = dp.shuffle(buffer_size=10)
output = list(shuffle_dp)

dataloader = dataloader2.DataLoader2(shuffle_dp, datapipe_adapter_fn=dataloader2.adapter.Shuffle(enable=False))

for data in dataloader:
    print(data)
