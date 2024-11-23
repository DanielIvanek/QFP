import numpy as np
import numpy.random as rn

rn.seed(100)
for i in range(1,8):
    random = rn.randn(10**i)
    print(f"Random numbers:{10**i:9.0f}  Mean: {np.mean(random):.12f}. Standard deviation: {np.std(random):.12f}.")
         
    
    
    
rn.seed(100)
for i in range(1,8):
    random = rn.randn(round(10**i/2))
    random=np.concatenate((random, -random))
    print(f"Random numbers:{10**i:9.0f}  Mean: {np.mean(random):.12f}. Standard deviation: {np.std(random):.12f}.")
         
    
    
rn.seed(100)
for i in range(1,8):
    random = rn.randn(round(10**i))
    random = (random - np.mean(random)) / np.std(random)
    print(f"Random numbers:{10**i:9.0f}  Mean: {np.mean(random):.12f}. Standard deviation: {np.std(random):.12f}.")
         
    
