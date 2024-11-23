import numpy.random as rn 
import matplotlib.pyplot as plt

rn.seed(100)  # fixes the seed for the reproducibility

rnu = rn.rand(100000,1) # generate the list with 100000x1 pseudorandom numbers from uniform (0,1) distribution
rnn = rn.randn(100000,1) # generate the list with 100000x1 pseudorandom numbers from standard normal distribution
rn1 = rn.beta(2, 5, (100000, 1)) # generate 100,000 pseudorandom numbers from beta(2,5) distribution
rn2 = rn.binomial(20, 0.5, (100000, 1)) # generate 100,000 pseudorandom numbers from binomial(20,0.5) distribution
rn3 = rn.chisquare(4, (100000, 1)) # generate 100,000 pseudorandom numbers from chisquare(4) distribution
rn4 = rn.exponential(1, (100000, 1)) # generate 100,000 pseudorandom numbers from exponential(1) distribution
rn5 = rn.lognormal(0,0.5, (100000, 1)) # generate 100,000 pseudorandom numbers from lognormal(0,0.5) distribution
rn6 = rn.normal(0,0.5, (100000, 1)) # generate 100,000 pseudorandom numbers from lognormal(0,0.5) distribution
rn7 = rn.uniform(0,1, (100000, 1)) # generate 100,000 pseudorandom numbers from uniform(0,1) distribution

plt.figure(figsize=(10,6))
plt.hist(x=rn6, bins=100, density=True)
plt.xlabel('Value')
plt.ylabel('Probability')
plt.title('Histogram of pseudorandom numbers')
plt.grid(False)
plt.show()
