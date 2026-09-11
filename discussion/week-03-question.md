---
id: w03-jyan36-ridge-stability
title: "Why Ridge Improves Stability"
author: jyan36
---

From the ridge solution

$$
\hat\beta_\lambda=(X^TX+n\lambda I)^{-1}X^T\widetilde y,
$$

why does adding $n\lambda I$ improve stability? What do $n$, $\lambda$, and $I$ each represent, and why is the identity matrix used? How does this relate to nearly collinear predictors or small singular values of $X$?