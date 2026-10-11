---
id: w07-jyan36-test-versus-cv-error
title: "Why Can Test Error Be Much Lower Than Cross-Validation Error?"
author: jyan36
---

In my LDA/QDA analysis of the SAheart dataset, the selected LDA model achieved a mean cross-validation error of 29.43%, but its error on a reserved test set of 93 observations was only 17.20%. The test set was not used for model selection. Does this large difference suggest a problem with the validation procedure, or could it reflect sampling variability and differences in training sample size? What checks would help distinguish these explanations?
