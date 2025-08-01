sample_tsne_012_subset <- read.csv('tsne_sample012_subset.csv')
sample_tsne_034_subset <- read.csv('tsne_sample034_subset.csv')
View(sample012_rsec_mols)
sample01234_rsec_mols <- merge(sample012_rsec_mols,sample034_rsec_mols,by = "Cell_Index")
View(sample034_rsec_mols)
sample012_rsec_mols$sample <- "wildtype"
View(sample012_rsec_mols)
View(sample012_rsec_mols)
sample034_rsec_mols$sample <- "gmcsf"
sample01234_rsec_mols <- cbind(sample012_rsec_mols,sample034_rsec_mols)
sample01234_rsec_mols <- rbind(sample012_rsec_mols,sample034_rsec_mols)
View(sample01234_rsec_mols)
sample012_sum <- colSums(sample012_rsec_mols, na.rm=TRUE)
sample012_sum <- colSums(sample012_rsec_mols[,2:404], na.rm=TRUE)
sample012_sum <- data.frame(colSums(sample012_rsec_mols[,2:404], na.rm=TRUE))
View(sample012_sum)
sample034_sum <- data.frame(colSums(sample034_rsec_mols[,2:404], na.rm=TRUE))
View(sample034_sum)
colnames(sample012_sum)[1] <- "total_count"
colnames(sample012_sum)[1] <- "total_count_12"
colnames(sample034_sum)[1] <- "total_count_34"
Fold_change <- data.frame(sample012_sum$total_count_12/sample034_sum$total_count_34)
View(Fold_change)
Fold_change <- data.frame(sample012_sum$total_count_12/sample034_sum$total_count_34)
results <- data.frame(names(sample012_sum),Fold_change$sample012_sum.total_count_12.sample034_sum.total_count_34)
View(results)
View(sample012_sum)
results <- data.frame(rownames(sample012_sum),Fold_change$sample012_sum.total_count_12.sample034_sum.total_count_34)
View(results)
colnames(results)[1] <- "IDs"
colnames(results)[2] <- "FoldChangevalues"
Fold_change <- data.frame(sample034_sum$total_count_34/sample012_sum$total_count_12)
results <- data.frame(rownames(sample012_sum),Fold_change$sample034_sum.total_count_34.sample012_sum.total_count_12)
colnames(results)[1] <- "IDs"
colnames(results)[2] <- "FoldChangevalues"
write.csv(results,"Fold_Changevalues_sample12_vs_sample34.csv")
results$log2FoldChangevalues <- log2(results$FoldChangevalues)
write.csv(results,"Fold_Changevalues_sample12_vs_sample34.csv")
