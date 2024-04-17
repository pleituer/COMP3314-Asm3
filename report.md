## COMP3314 Asm 3 

### Background and brief analysis of dataset

### Models

#### Model 1 (Based on HOG features)

#### Model 2 (Based on K-Means feature learning)

#### Model 3 (Based on Sparse Encoding and a more generalized K-Means algorithm for feature learning)

### Methodology 

#### Fine-tuning

Instead of using Grid Search, we propose to use a faster algorithm adapted from Numerical Optimization, the Golden Section Method. As we notice that using SVM, we only have one parameter to tune, which is $C$, and we hypothesis that the function $f(C)$, which gives the mean accuracy for some $C$, is concave. Hence, adapting from the Golden Section Method, we propose the following algorithm to find tune: 

Given a target precision of $\epsilon$, and an intial interval $[a_1, b_1]$, where we know the optimal hyperparameter exists in, we let $\phi = \frac{\sqrt{5}-1}{2}$, then let $$\begin{aligned} \lambda_1 &= a_1 + (1-\phi)(b-a) \\ \mu_1 &= a_1 + \phi(b-a)  \end{aligned}$$ Then, we execute the following until the target precision is reached, i.e. $b_k - a_k < \epsilon$. 
- If $f(\lambda_k) > f(\mu_k)$, then we set $a_{k+1} = a_k$, $b_{k+1} = \mu_k$, $\lambda_{k+1} = a_{k+1} + (1-\phi)(b_{k+1}-a_{k+1})$, and $\mu_{k+1} = \lambda_k$.
- Otherwise, we set $a_{k+1} = \lambda_k$, $b_{k+1} = b_k$, $\lambda_{k+1} = \mu_k$, and $\mu_{k+1} = a_{k+1} + \phi(b_{k+1}-a_{k+1})$. 

After iterating, we let $C^* = \frac{a_n+b_n}{2}$ to be the optimal parameter. As by induction, it is quite easy to show that $\frac{b_k - a_k}{b_{k+1} - a_{k+1}} = \phi$ for any $1 \leq k \leq n-1$, hence to achieve a certain accuracy $\epsilon$, $1 + \log _{\phi} \frac{\epsilon}{b_1 - a_1}$ iterations is sufficient. 

However, we acknowledge that the parameter-to-accuracy curve is not concave, especially around the maximum, hence we use this algorithm to shrink our search to a small enough interval, then perform Grid Search on that interval.