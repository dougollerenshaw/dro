import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

savepath = "/allen/programs/braintv/workgroups/nc-ophys/Doug/2020.03.26_temp"
fn = str(datetime.datetime.now()).replace(' ','_').replace(':','_')+'.png'

fig,ax=plt.subplots()
ax.plot(np.random.randn(1000))
ax.set_title(fn)
print('saving {}'.format(os.path.join(savepath, fn)))
fig.savefig(os.path.join(savepath, fn))