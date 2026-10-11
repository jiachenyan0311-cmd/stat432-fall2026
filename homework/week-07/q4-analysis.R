# From homework/week-07: Rscript q4-analysis.R train
# Review the training/CV results, then record the review in R:
# writeLines("Training/CV reviewed; selection frozen.", "q4-output/training-review.txt")
# Finally: Rscript q4-analysis.R test
library(MASS)
seed <- 43207
out <- "q4-output"
dir.create(out, showWarnings = FALSE)

folds <- function(y) {
  f <- integer(length(y))
  for (k in levels(y)) {
    ids <- which(y == k)
    f[ids] <- sample(rep(1:5, length.out = length(ids)))
  }
  f
}

cv_errors <- function(X, y, fold_sets) {
  t(sapply(fold_sets, function(f) {
    pred <- matrix(NA_character_, nrow(X), 2)
    for (j in 1:5) {
      fit_lda <- lda(X[f != j, , drop = FALSE], y[f != j])
      fit_qda <- qda(X[f != j, , drop = FALSE], y[f != j])
      pred[f == j, 1] <- as.character(predict(fit_lda, X[f == j, , drop = FALSE])$class)
      pred[f == j, 2] <- as.character(predict(fit_qda, X[f == j, , drop = FALSE])$class)
    }
    c(LDA = mean(pred[, 1] != y), QDA = mean(pred[, 2] != y))
  }))
}

train_analysis <- function() {
  if (file.exists(file.path(out, "final-results.rds"))) {
    stop("Final test results already exist; reuse them without revising selection.")
  }
  set.seed(seed)
  heart <- read.csv("data/saheart.csv")
  heart$famhist <- factor(heart$famhist, levels = c("Absent", "Present"))
  X <- data.matrix(heart[, names(heart) != "chd"])
  y <- factor(heart$chd)
  stopifnot(ncol(X) == 9, all(is.finite(X)), !anyNA(y))

  train_ids <- unlist(lapply(levels(y), function(k) {
    ids <- which(y == k)
    sample(ids, floor(0.8 * length(ids)))
  }))
  test_ids <- setdiff(seq_len(nrow(X)), train_ids)
  Xt <- X[train_ids, , drop = FALSE]
  yt <- y[train_ids]
  fold_sets <- replicate(10, folds(yt), simplify = FALSE)
  errors <- cv_errors(Xt, yt, fold_sets)
  avg <- colMeans(errors)
  selected <- names(avg)[which.min(avg)]
  wins <- c(LDA = sum(errors[, 1] < errors[, 2]),
            QDA = sum(errors[, 2] < errors[, 1]),
            Ties = sum(errors[, 1] == errors[, 2]))

  placebo <- replicate(50, {
    shuffled <- sample(yt)
    shuffled_folds <- replicate(10, folds(shuffled), simplify = FALSE)
    min(colMeans(cv_errors(Xt, shuffled, shuffled_folds)))
  })
  majority_error <- 1 - max(prop.table(table(yt)))
  summary <- data.frame(Model = names(avg), CV_error = avg,
                        Split_SD = apply(errors, 2, sd))
  write.csv(summary, file.path(out, "cv-summary.csv"), row.names = FALSE)
  write.csv(data.frame(errors), file.path(out, "cv-repetitions.csv"), row.names = FALSE)
  write.csv(data.frame(Error = placebo), file.path(out, "placebo-errors.csv"), row.names = FALSE)
  results <- list(seed = seed, X = X, y = y, train_ids = train_ids,
                  test_ids = test_ids, fold_sets = fold_sets, errors = errors,
                  summary = summary, selected = selected, wins = wins,
                  placebo = placebo, majority_error = majority_error)
  saveRDS(results, file.path(out, "training-results.rds"))
  print(summary)
  print(wins)
  cat("Selected:", selected, "\nTraining:", length(train_ids),
      " Test reserved:", length(test_ids), "\nMajority error:", majority_error,
      "\nPlacebo mean:", mean(placebo), "\n")
}

test_analysis <- function() {
  if (file.exists(file.path(out, "final-results.rds"))) {
    stop("Test was already evaluated. Render from saved results.")
  }
  r <- readRDS(file.path(out, "training-results.rds"))
  stopifnot(file.exists(file.path(out, "training-review.txt")))
  fit_fun <- if (r$selected == "LDA") lda else qda
  fit <- fit_fun(r$X[r$train_ids, , drop = FALSE], r$y[r$train_ids])
  pred <- predict(fit, r$X[r$test_ids, , drop = FALSE])$class
  r$confusion <- table(Observed = r$y[r$test_ids], Predicted = pred)
  r$test_errors <- sum(pred != r$y[r$test_ids])
  n <- length(r$test_ids)
  p <- r$test_errors / n
  z <- qnorm(0.975)
  center <- (p + z^2 / (2 * n)) / (1 + z^2 / n)
  half <- z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / (1 + z^2 / n)
  r$test_error <- p
  r$interval <- c(center - half, center + half)
  r$fit <- fit
  saveRDS(r, file.path(out, "final-results.rds"))
  print(r$confusion)
  cat("Selected:", r$selected, " Test error:", p,
      " Wilson interval:", r$interval, "\n")
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) && args[1] == "train") train_analysis()
if (length(args) && args[1] == "test") test_analysis()
