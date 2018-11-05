probas_best_model <- read.table(file = '/home/alessio/tesi/notebook/primo_esperimento/test_probas.csv', sep = ',', header = FALSE, row.names=NULL)
ones=rep(1, 40)
zeroes_count = nrow(probas_best_model) - 40
zeros = rep(0, zeroes_count)
test_y  = c(ones,zeros)
print(test_y)

library(precrec)
library(ggplot2)
curve <- evalmod(scores = probas_best_model$V1, labels = test_y)
autoplot(curve)
curve.part <- part(curve, xlim = c(0.0, 1))

paucs.df <- pauc(curve.part)
knitr::kable(paucs.df)

library(PRROC)
pr<-pr.curve(scores.class0 = head(probas_best_model$V1, 40), scores.class1 = tail(x = probas_best_model$V1, -40), curve = TRUE)
plot(pr)
