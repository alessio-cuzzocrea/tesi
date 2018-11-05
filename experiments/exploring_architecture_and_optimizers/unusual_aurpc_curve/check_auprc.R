probas <- read.table(file = '/home/alessio/tesi/experiments/exploring_architecture_and_optimizers/unusual_aurpc_curve/unusual_train_probas.csv',
                                sep = ',',
                                header = FALSE, 
                                row.names=NULL)
ones=rep(1, 356)
zeroes_count = nrow(probas) - 356
zeros = rep(0, zeroes_count)
y_true  = c(ones,zeros)
print(y_true)

library(precrec)
library(ggplot2)
curve <- evalmod(scores = probas$V1, labels = y_true)
autoplot(curve)
print(auc(curve))
# library(PRROC)
# pr<-pr.curve(scores.class0 = head(probas_best_model$V1, 40), scores.class1 = tail(x = probas_best_model$V1, -40), curve = TRUE)
# plot(pr)
