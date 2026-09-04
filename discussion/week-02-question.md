---
id: w02-jyan36-path-robustness
title: "Robustness versus reproducibility in AI-generated code"
author: jyan36
---

While working on Question 4, I needed to read `diabetes.csv`. An AI-generated solution tried several possible file paths and automatically selected one that existed. After checking the intended repository structure, I found that a single relative path was sufficient.

The fallback approach was more robust across different environments, but it also made the code harder to interpret and could hide an unresolved working-directory or repository-structure problem. A single explicit path is clearer, but it may fail when the environment changes.

From both the user and AI-agent design perspectives, how should we balance robustness, reproducibility, and transparency? When should an AI agent adapt to multiple possible environments, and when should it instead fail clearly and require the project structure to be fixed?
