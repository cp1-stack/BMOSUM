install.packages('ecp')

data("ACGH", package = "ecp")
acghData <- ACGH$data
acghData[,1]

plot(acghData[,1])

write.table(acghData,"acghData.csv",row.names=FALSE,col.names=TRUE,sep=",")














