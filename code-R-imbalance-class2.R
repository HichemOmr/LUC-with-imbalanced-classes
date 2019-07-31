# install.packages("caret")
# use mlbench, caret and DT library

install.packages(
  
  c("ggplot2"),
  repos = c("http://rstudio.org/_packages",
            "http://cran.rstudio.com")
)

#> You may also find it useful to restart R,

#> In RStudio, that's the menu Session >> Restart R

library(ggplot2)
library(caret)

install.packages(c("mlbench", "caret", "DT", "ggplot2"))
require(mlbench)
require(caret)
require(DT)

# load iris set
data(iris)
dim(iris)
m <- c("rf", "nnet", "svmRadialSigma")

# show which libraries were loaded 

sessionInfo()

# load X and Y (this will be transferred to to train function)

X = iris[c(1:50, 51:61, 100:110),1:3]
Y = iris[c(1:50, 51:61, 100:110),5]

inTrain <- createDataPartition(iris$Species)
training <- iris[inTrain[[1]], ] 
testing <- iris[-inTrain[[1]], ]

X_test = iris[,1:3] 
Y_test = iris[,5] 

K=3
w = (length(Y)/K) * (c(1/length(Y[Y=="setosa"]), 1/length(Y[Y=="versicolor"]), 1/length(Y[Y=="virginica"])))
weights = c(
  rep( w[1],length(Y[Y=="setosa"]) ),
  rep( w[2],length(Y[Y=="versicolor"]) ),
  rep( w[3],length(Y[Y=="virginica"]) )
)

# register parallel front-end

library(iterators)
library(doParallel)

cl <- makeCluster(detectCores()); registerDoParallel(cl)

trainCall <- function(i)
  
{
  cat("----------------------------------------------------","\n");
  set.seed(123); cat(i," <- loaded\n");
  
    model       = train(y=Y, x=X, (i), trControl = trainControl(method = "boot632"), weights = weights)
    predictions =  predict(model, newdata=X_test)
    cm          = confusion.matrix(predictions, Y_test)
    acc         = cm$accuracy
  return(acc)
}

# use lapply/loop to run everything, required for try/catch error function to work

t2 <- lapply(m, trainCall)



#remove NULL values, we only allow succesful methods, provenance is deleted.

t2 <- t2[!sapply(t2, is.null)]



# this setup extracts the results with minimal error handling

# TrainKappa can be sometimes zero, but Accuracy SD can be still available

# see Kappa value http://epiville.ccnmtl.columbia.edu/popup/how_to_calculate_kappa.html

printCall <- function(i)
  
{
  return(tryCatch(
    {
      cat(sprintf("%-22s",(m[i])))
      
      cat(round(getTrainPerf(t2[[i]])$TrainAccuracy,4),"\t")
      
      cat(round(getTrainPerf(t2[[i]])$TrainKappa,4),"\t")
      
      cat(t2[[i]]$times$everything[3],"\n")},
    
    error=function(e) NULL))
  
}



r2 <- lapply(1:length(t2), printCall)



# stop cluster and register sequntial front end

stopCluster(cl); registerDoSEQ();



# preallocate data types

i = 1; MAX = length(t2);

Name <- character() # Name

Accuracy <- numeric()   # R2

Kappa <- numeric()      # RMSE

Time <- numeric()       # time [s]

Description <- character()       # long model name



# fill data and check indexes and NA with loop/lapply

for (i in 1:length(t2)) {
  
  Name[i] <- t2[[i]]$method
  
  Accuracy[i] <- as.numeric(round(getTrainPerf(t2[[i]])$TrainAccuracy,4))
  
  Kappa[i] <- as.numeric(round(getTrainPerf(t2[[i]])$TrainKappa,4))
  
  Time[i] <- as.numeric(t2[[i]]$times$everything[3])
  
  Description[i] <- t2[[i]]$modelInfo$label
  
}



# coerce to data frame

df1 <- data.frame(Name,Accuracy,Kappa,Time,Description, stringsAsFactors=FALSE)



# print all results to R-GUI

df1



# plot models, just as example

# ggplot(t2[[1]])

# ggplot(t2[[1]])



# call web output with correct column names

datatable(df1,  options = list(
  
  columnDefs = list(list(className = 'dt-left', targets = c(0,1,2,3,4,5))),
  
  pageLength = MAX,
  
  order = list(list(2, 'desc'))),
  
  colnames = c('Num', 'Name', 'Accuracy', 'Kappa', 'time [s]', 'Model name'),
  
  caption = paste('Classification results from caret models',Sys.time()),
  
  class = 'cell-border stripe')  %>%                     
  
  formatRound('Accuracy', 3) %>% 
  
  formatRound('Kappa', 3) %>%
  
  formatRound('Time', 3) %>%
  
  formatStyle(2,
              
              background = styleColorBar(x2, 'steelblue'),
              
              backgroundSize = '100% 90%',
              
              backgroundRepeat = 'no-repeat',
              
              backgroundPosition = 'center'
              
  )



# print confusion matrix example

caret::confusionMatrix(t2[[1]])





### END

