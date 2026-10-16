# Explicit invocation: follow the adjacent SKILL.md on ../data/saheart.csv.
# Response: chd; all nine predictors; famhist: Absent=1, Present=2.
# Run from this directory: Rscript analysis.R
library(MASS)
seed <- 43208
set.seed(seed)
d <- read.csv("../data/saheart.csv")
stopifnot(all(d$chd %in% c(0, 1)),
          all(d$famhist %in% c("Absent", "Present")))
d$famhist <- match(d$famhist, c("Absent", "Present"))
X <- as.matrix(d[, setdiff(names(d), "chd")])
y <- factor(d$chd, levels = c(0, 1))
stopifnot(ncol(X) == 9, all(is.finite(X)), !anyNA(y))
train <- sort(unlist(lapply(levels(y), function(k) {
  ids <- which(y == k)
  sample(ids, floor(0.7 * length(ids)))
})))
test <- setdiff(seq_len(nrow(d)), train)
Xt <- X[train, , drop = FALSE]
yt <- y[train]
fold <- integer(length(train))
for (k in levels(yt)) {
  ids <- which(yt == k)
  fold[ids] <- sample(rep(1:5, length.out = length(ids)))
}
write.csv(data.frame(observation = seq_len(nrow(d)),
                     split = ifelse(seq_len(nrow(d)) %in% train, "train", "test")),
          "split.csv", row.names = FALSE)
write.csv(data.frame(observation = train, fold = fold), "folds.csv", row.names = FALSE)
fits <- list(LDA = MASS::lda, QDA = MASS::qda)
errors <- matrix(NA_real_, 5, 2, dimnames = list(1:5, names(fits)))
failures <- character()
ranks <- list()
for (j in 1:5) {
  a <- fold != j
  z <- scale(Xt[a, , drop = FALSE])
  v <- scale(Xt[!a, , drop = FALSE], center = attr(z, "scaled:center"),
             scale = attr(z, "scaled:scale"))
  ranks[[j]] <- sapply(levels(yt), function(k) qr(cov(z[yt[a] == k, ]))$rank)
  for (name in names(fits)) {
    tryCatch({
      fit <- fits[[name]](z, yt[a])
      pred <- predict(fit, v)$class
      errors[j, name] <- mean(pred != yt[!a])
    }, error = function(e) {
      failures <<- c(failures, sprintf("Fold %s, %s: %s", j, name, conditionMessage(e)))
    })
  }
}
writeLines(if (length(failures)) failures else "No fitting failures.", "fit-log.txt")
write.csv(errors, "cv-fold-errors.csv")
if (anyNA(errors)) stop("See fit-log.txt; failed candidates require review.")
cv <- colMeans(errors)
# LDA precedes QDA, so exact ties favor the simpler covariance model.
selected <- names(which.min(cv))
z <- scale(Xt)
final <- fits[[selected]](z, yt)
# Model selection is now complete. Evaluate the reserved test set once.
v <- scale(X[test, , drop = FALSE], center = attr(z, "scaled:center"),
           scale = attr(z, "scaled:scale"))
pred <- predict(final, v)$class
cm <- table(Observed = y[test], Predicted = pred)
test_error <- mean(pred != y[test])
summary <- data.frame(Model = names(cv), Setting = "All 9 predictors",
                      Mean_CV_Error = as.numeric(cv))
write.csv(summary, "cv-summary.csv", row.names = FALSE)
write.csv(cm, "test-confusion.csv", row.names = FALSE)
saveRDS(list(seed = seed, train = train, test = test, fold = fold,
             cv = errors, selected = selected, fit = final,
             center = attr(z, "scaled:center"), scale = attr(z, "scaled:scale"),
             confusion = cm, test_error = test_error, ranks = ranks), "results.rds")

# A fixed, single-page PDF report; code and detailed outputs are separate.
pdf("analysis-report.pdf", width = 8.5, height = 11)
layout(matrix(1:4, ncol = 1), heights = c(1.25, 1.05, 2.3, 2.1))
par(mar = c(0.2, 0.8, 0.3, 0.8))
panel <- function(lines, size = 0.95) {
  plot.new()
  text(0, seq(0.92, 0.08, length.out = length(lines)), lines,
       adj = c(0, 0.5), cex = size)
}
panel(c("SAheart: LDA/QDA Classification Analysis",
        sprintf("Seed %s; response chd; 9 predictors; famhist: Absent=1, Present=2.", seed),
        sprintf("462 complete cases (302/160 by class); no rows or features excluded."),
        sprintf("Stratified split: %s training (%s/%s), %s test (%s/%s).", length(train),
                sum(yt == 0), sum(yt == 1), length(test), sum(y[test] == 0), sum(y[test] == 1)),
        "Identical stratified 5-fold CV; scaling fitted within each training fold."))
panel(c("Candidate                 Setting                    Mean CV error",
        sprintf("LDA                       All 9 predictors           %.4f", cv["LDA"]),
        sprintf("QDA                       All 9 predictors           %.4f", cv["QDA"]),
        "Both mandatory models fit successfully; all class covariance ranks were 9."))
par(mar = c(3, 4, 1, 1))
matplot(1:5, errors, type = "b", pch = c(16, 17), lty = c(1, 2),
        col = c("#2457A7", "#B34032"), xaxt = "n", xlab = "Validation fold",
        ylab = "Misclassification rate", ylim = range(c(errors, cv)))
axis(1, at = 1:5)
abline(h = cv, col = c("#2457A7", "#B34032"), lty = 3)
legend("topright", names(cv), col = c("#2457A7", "#B34032"),
       pch = c(16, 17), lty = c(1, 2), bty = "n")
par(mar = c(0.2, 0.8, 0.4, 0.8))
panel(c(sprintf("Selected by mean CV error: %s; refitted on all %s training cases.", selected, length(train)),
        sprintf("One held-out evaluation: error %.4f (%s/%s); accuracy %.4f.", test_error,
                sum(pred != y[test]), length(test), 1 - test_error),
        "Confusion matrix: observed rows / predicted columns (0, 1)",
        sprintf("Observed 0:   %s   %s", cm[1, 1], cm[1, 2]),
        sprintf("Observed 1:   %s   %s", cm[2, 1], cm[2, 2]),
        sprintf("Class-1 sensitivity %.3f; specificity %.3f.", cm[2, 2]/sum(cm[2, ]), cm[1, 1]/sum(cm[1, ])),
        "LDA estimates a shared covariance; QDA estimates separate class covariances.",
        "Class imbalance and finite samples limit precision; QDA is more parameter intensive.",
        "Full covariance rank does not guarantee stable estimates. One split and five folds",
        "give uncertain performance estimates; results compare only these two candidates",
        "and do not establish universal superiority."), size = 0.88)
dev.off()
print(summary)
cat("Selected:", selected, "\nTest error:", test_error, "\n")
print(cm)
