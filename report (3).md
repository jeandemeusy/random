# Report - Deep Learning - Session 3

## Exercice 2

### a)

$$\frac{1}{2m} \sum_{i=1}^m \left(y^{(i)} - \left(\sum_{k=1}^n w_{2,k} \sigma(w_{1,k}x^{(i)} + b_{1,k}) + b_2 \right)  \right)^2$$

Where:

* m: batch size
* n: number of neurons in the hidden layer

Computing the derivative:

For simplification purposes we will compute the derivative for a single pair (x, y). We also introduce some intermediate variables.

$$J_{\text{MSE}}(E) = \frac{1}{2}E^2$$

$$E(y, \hat{y}) = y - \hat{y}$$

$$\hat{y}(w_2, z_1, b_2) = \sum_{k=1}^n w_{2,k} z_{1,k} + b_2$$

$$z_{1,k}(out_{1,k}) = \frac{1}{1 + e^{-out_{1,k}}}$$

$$out_{1,k}(w_{1,k}, x, b_{1,k}) = w_{1,k} x + b_{1,k}$$

Now we can use the chain rule with more ease.

$$\frac{\partial J_{MSE}(E)}{\partial E} = E$$

$$\frac{\partial E(y, \hat{y})}{\partial \hat{y}} = -1$$

$$\frac{\partial \hat{y}(w_2, z_1, b_2)}{\partial w_{2,k}} = z_{1,k}$$

We get the derivative for the weights in the last layer.

$$\frac{\partial J_{MSE}(E)}{\partial w_{2,k}} = \frac{\partial J_{MSE}(E)}{\partial E} \frac{\partial E(y, \hat{y})}{\partial \hat{y}} \frac{\partial \hat{y}(w_2, z_1, b_2)}{\partial w_{2,k}} = -Ez_{1,k} = -(y - \hat{y})z_{1,k}$$

The partial derivative for a given batch would be:

$$\frac{\partial J_{MSE}(E)}{\partial w_{2,k}} = - \frac{1}{m} \sum_{i=1}^m (y^{(i)} - \hat{y^{(i)}}) z_{1,k}^{(i)}$$

For the bias in the last layer.

$$\frac{\partial J_{MSE}(E)}{\partial b_2} = \frac{\partial J_{MSE}(E)}{\partial E} \frac{\partial E(y, \hat{y})}{\partial \hat{y}} \frac{\partial \hat{y}(w_2, z_1, b_2)}{\partial b_2} = -(y - \hat{y})$$

The partial derivative for a given batch would be:

$$\frac{\partial J_{MSE}(E)}{\partial b_2} = - \frac{1}{m} \sum_{i=1}^m (y^{(i)} - \hat{y^{(i)}})$$

The derivative for the weights in the first layer is given by

$$\frac{\partial J_{MSE}(E)}{\partial w_{2,k}} = \frac{\partial J_{MSE}(E)}{\partial E} \frac{\partial E(y, \hat{y})}{\partial \hat{y}} \frac{\partial \hat{y}(w_2, z_1, b_2)}{\partial z_1} \frac{\partial z_1(out)}{\partial out} \frac{\partial out(x, b_1, w_{1,k})}{\partial w_{1,k}} = -(y - \hat{y}) w_{2,k} z_{1,k}(1 - z_{1,k})x$$

The partial derivative for a given batch would be:

$$\frac{\partial J_{MSE}(E)}{\partial w_{2,k}} = -\frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)}) w_{2,k} z_{1,k}^{(i)}(1 - z_{1,k}^{(i)})x^{(i)}$$

In matrix notation, for a quicker implementation in numpy:

* y in R(m, 1)
* y_hat in R(m, 1)
* w2 in R(1, n)
* z1 in R(m, n)
* x in R(m, 1)
* 1 in R(1, m)

$$\frac{1}{m} \vec{x}^T (\vec{y} - \vec{\hat{y}}) \mathbf{1} \vec{z_1}' \odot \vec{w_2}$$

The derivative for the bias in the first layer is given by

$$\frac{\partial J_{MSE}(E)}{\partial w_{2,k}} = \frac{\partial J_{MSE}(E)}{\partial E} \frac{\partial E(y, \hat{y})}{\partial \hat{y}} \frac{\partial \hat{y}(w_2, z_1, b_2)}{\partial z_1} \frac{\partial z_1(out)}{\partial out} \frac{\partial out(x, b_1, w_{1,k})}{\partial b_1} = -(y - \hat{y}) w_{2,k} z_{1,k}(1 - z_{1,k})$$

The partial derivative for a given batch would be:

$$\frac{\partial J_{MSE}(E)}{\partial w_{2,k}} = -\frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)}) w_{2,k} z_{1,k}^{(i)}(1 - z_{1,k}^{(i)})$$


<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script>