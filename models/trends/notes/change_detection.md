### Suppose we have the following: 

$$x_i = \text{Observed value at time } t$$

$$\mu = \text{Mean of x if no change}$$

$$S_t = max(0, S_{t-1} + (x_t - \mu - C))$$

$$S_t ≥ T$$

Where T is a defined threshold and $S_t$ is the maximum value between 0 and the sum of the previous observed value at time $t -1$

Thus, we need to define a threshold $T$ and a critical level $C$, we can check for an **Increase** in deviations from the mean. 

To check for an increase, $(x_t - \mu - C)$; for a decrease $(\mu - x_t - C)$, and both ways can be checked with $(|x_t - \mu| - C)$
The algorithim will go as follows: 

1. Calculate the mean of the daily averages
2. Calculate $x_t - \mu - C$
3. Calculate $max(0, S_{t-1} + (x_t - \mu - C))$ (Depending on which direction you wish to look for a change.)

### Things to keep in mind about paramaters $C$ & $T$

1. The larger the value for $C$, the harder it is for $S_t$ to get large; thus the model will be __Less__ sensitive to achieve. 
2. The smaller the value for $C$, the *easier* it is for $S_t$ to get large, thus the model will be __more__ sensitive to sudden changes. 

Thus its important for us to find an *optimal* parameter for $C$, where the change is not so easily or difficult to be detected. In other words we want the change to be detected no more then $M$ times, and no less than $N$ times. Lets first have a sanity check, and _manually_ test for different values of $C$: