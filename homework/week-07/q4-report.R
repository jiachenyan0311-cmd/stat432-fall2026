r <- readRDS("q4-output/final-results.rds")
pct <- function(x) sprintf("%.2f%%", 100 * x)

draw_report <- function() {
  navy <- "#142B49"
  teal <- "#007F86"
  magenta <- "#B31557"
  ink <- "#172638"
  par(mar = c(0, 0, 0, 0), family = "sans", fg = ink)
  plot.new()
  plot.window(xlim = c(0, 1), ylim = c(0, 1), xaxs = "i", yaxs = "i")

  label <- function(x, y, value, size = 1.05, color = ink, bold = FALSE, adj = 0) {
    text(x, y, value, adj = c(adj, 1), cex = size,
         col = color, font = if (bold) 2 else 1)
  }
  paragraph <- function(y, value, width = 91, size = 1.05) {
    lines <- strwrap(value, width = width)
    for (i in seq_along(lines)) label(0.07, y - (i - 1) * 0.022, lines[i], size)
  }

  rect(0.07, 0.975, 0.93, 0.982, col = teal, border = NA)
  label(0.07, 0.955, "LDA or QDA?", size = 2, color = navy, bold = TRUE)
  label(0.07, 0.910, "SAheart | Prediction, stability & shuffled-label checks", size = 1.1)
  label(0.07, 0.877,
        sprintf("Seed %d   |   369 training / 93 test   |   9 predictors", r$seed),
        size = 1.05, color = teal, bold = TRUE)

  label(0.07, 0.838, "METHOD", size = 1.1, color = navy, bold = TRUE)
  paragraph(0.810, paste(
    "Predict CHD; famhist is coded Absent=1, Present=2. Compare ordinary LDA and QDA",
    "with shared stratified 5-fold CV, repeated 10 times. Each fit estimates its own",
    "priors, means and covariances; no learned preprocessing. Select the lowest mean",
    "error (ties: LDA). Reserve a stratified 20% test set before selection.",
    "Shuffle training labels 50 times and repeat the full CV comparison each time."
  ))

  label(0.07, 0.669, "CROSS-VALIDATION", size = 1.1, color = navy, bold = TRUE)
  rect(0.07, 0.609, 0.93, 0.640, col = navy, border = NA)
  xx <- c(0.095, 0.48, 0.73)
  for (i in 1:3) label(xx[i], 0.632, c("Model", "Mean error", "Split SD (pp)")[i],
                      size = 1.05, color = "white", bold = TRUE)
  for (i in 1:2) {
    top <- 0.609 - (i - 1) * 0.033
    rect(0.07, top - 0.033, 0.93, top,
         col = if (i == 1) "#E4F3F3" else "#F0F3F7", border = NA)
    values <- c(r$summary$Model[i], pct(r$summary$CV_error[i]),
                sprintf("%.2f", 100 * r$summary$Split_SD[i]))
    for (j in 1:3) label(xx[j], top - 0.008, values[j],
                        size = 1.1, bold = i == 1)
  }
  label(0.07, 0.535, "Repeat wins: LDA 8 / QDA 1 / ties 1. Selected: LDA.", size = 1.05)

  label(0.07, 0.493, "PERMUTATION CHECK", size = 1.1, color = navy, bold = TRUE)
  label(0.07, 0.268,
        sprintf("Shuffled-label mean: %s. All 50 errors exceeded observed CV error.",
                pct(mean(r$placebo))), size = 1.02)

  rect(0.07, 0.176, 0.93, 0.228, col = navy, border = NA)
  label(0.095, 0.215, "TEST ERROR", size = 0.95, color = "white", bold = TRUE)
  label(0.30, 0.216, pct(r$test_error), size = 1.7, color = "white", bold = TRUE)
  label(0.53, 0.214, "16 / 93 errors; evaluated once", size = 1, color = "white")
  label(0.53, 0.193, "95% Wilson interval: 10.88%-26.13%", size = 0.95, color = "white")

  label(0.07, 0.153, "FINDINGS & LIMITATIONS", size = 1.1, color = navy, bold = TRUE)
  paragraph(0.127, paste(
    "LDA had lower CV error and less variation across repeated splits; its shared",
    "covariance assumption may reduce estimation variance.",
    "Final fit: two 9-entry means, one 9 x 9 covariance; priors 0.653 / 0.347.",
    "The permutation comparison suggests predictive information, not causality.",
    "CV/test differences may reflect small-test sampling variability. Repeated CV",
    "folds overlap: split SD is not a confidence interval. Placebo is descriptive;",
    "clinical usefulness is unestablished."
  ))

  par(fig = c(0.07, 0.93, 0.289, 0.475), new = TRUE,
      mar = c(2.7, 3.0, 0.35, 0.2), mgp = c(1.6, 0.45, 0),
      col.axis = ink, col.lab = ink)
  observed <- 100 * min(r$summary$CV_error)
  h <- hist(100 * r$placebo, breaks = 10, plot = FALSE)
  plot(h, xlim = c(28.5, 39), ylim = c(0, max(h$counts) + 3),
       col = teal, border = "white", main = "", axes = FALSE,
       xlab = "Minimum repeated-CV error (%)", ylab = "Count", cex.lab = 1)
  axis(1, at = seq(29, 39, 2), cex.axis = 0.95)
  axis(2, las = 1, cex.axis = 0.9)
  abline(v = observed, col = magenta, lwd = 3)
  text(observed + 0.28, max(h$counts) * 0.61,
       paste0("Observed\n", pct(observed / 100)), adj = 0,
       col = magenta, font = 2, cex = 1.05)
  abline(v = 100 * r$majority_error, col = navy, lty = 2, lwd = 2)
  text(100 * r$majority_error - 0.18, max(h$counts) + 2,
       "Training majority-class baseline: 34.69%", adj = 0.5, col = navy, cex = 0.92)
}

pdf("q4-output/q4-report.pdf", width = 8.5, height = 11, family = "Helvetica")
draw_report()
dev.off()
png("q4-output/q4-report.png", width = 1700, height = 2200, res = 200)
draw_report()
dev.off()
