###########################################################################################
#################### TRABAJO FINAL DEL CURSO MACHINE LEARNING #############################
###########################################################################################

# NOMBRE: BRIAN LA ROSA LA ROSA

rm(list = ls())
dev.off()
options(scipen=999) 

#--------------------------------------------
# Paquetes
library(foreign)
library(gmodels)
library(partykit)
library(rpart)
library(rpart.plot)
library(caTools)
library(caret)
library(ggplot2)
library(MLmetrics)
library(randomForest)
library(ISLR)
library(DataExplorer)
library(zeallot)
library(VIM)
library(missForest)
library(dummies)
library(iplots)
library(rJava)


#######################
# 1. LECTURA DE DATOS #
#######################
# Cargando los datos 

datos<-read.csv("AdquisicionAhorro.csv",
                stringsAsFactors = T, 
                sep=",",
                na.strings = "")
str(datos)

# Eliminando la columna de identificacion de la persona (coddoc) ya que no aporta nada
# a la evaluacion

datos$coddoc <- NULL

# veamos si hay datos perdidos

library(DataExplorer)
windows()
plot_missing(datos)  # vemos que no hay ningun dato perdido, asi que podemos continuar


#####################################
### 2. Particion Muestral ###########
#####################################

## Particionando la Data
table(datos$Adq_Ahorro) # vemos que solo el 258/2240 = 0.115% del total de personas
set.seed(1234)          # adquirio o se ha susucrito a un deposito a plazo

library(caret)
sample <- createDataPartition(datos$Adq_Ahorro, 
                              p = .70,list = FALSE,times = 1)

data.train <- datos[ sample,] # Dataset de Entrenamiento
data.test <- datos[-sample,] # Dataset de Validacion


options(scipen=999)
windows()
boxplot(datos$balance ~ datos$Adq_Ahorro, 
        main= "BoxPlot de balance  vs Adq_Ahorro",
        xlab = "Cluster", 
        col = c("red","blue"))


options(scipen=999)
windows()
boxplot(datos$duracion ~ datos$Adq_Ahorro, 
        main= "BoxPlot de balance  vs Adq_Ahorro",
        xlab = "Cluster", 
        col = c("red","blue"))

#####################################
# 3. Modelamiento Predictivo ########
#####################################

###### XGBoost ###########

library(xgboost)

Mtrain_XGB <- model.matrix( ~., data=data.train[,c(1:12)]) # hasta la columna 12 ya que
                                                            # la ultima es la que quiero 
                                                            # hallar (Adq_Ahorro)
Ytrain <- as.vector(data.train$Adq_Ahorro)

# Construimos una matriz Xgboost
dtrain <- xgb.DMatrix(Mtrain_XGB, label = Ytrain)

###########################################################
# Lo mismo que le hago al train, le hago al test
Mtest_XGB <- model.matrix( ~., data=data.test[,c(1:12)])

Ytest <- as.vector(data.test$Adq_Ahorro)

dtest <- xgb.DMatrix(Mtest_XGB, label = Ytest)
##############################################################

#Hacemos nuestra lista de particiones de datos
watchlist <- list(train = dtrain, test = dtest)

# Escogemos los parametros de una manera apropiada
param <- list(booster = "gbtree", 
              objective = "binary:logistic", 
              eta=0.3,
              alpha=0.8,
              gamma=0.7,
              max_depth=3, 
              #min_child_weight=1, 
              subsample=0.6, 
              colsample_bytree=0.6,
              eval_metric = "auc")

xgb_fit <- xgb.train(param, 
                     dtrain, 
                     nround = 1000, 
                     watchlist,verbose = 1,
                     early_stopping_rounds = 15)


importance_xgb <- xgb.importance(feature_names = colnames(Mtrain_XGB),
                                 model = xgb_fit)
importance_xgb
# Plot de las variables
windows()
xgb.plot.importance(importance_matrix = importance_xgb)

# Prediccion
pred_xgb_t<- predict(xgb_fit,dtest,type="response")

# Convertimos la clase a probabilidad

clas_Xgb <- ifelse(pred_xgb_t<0.50,'0','1') # el 0.5 que coloque depende de que tanto 
clas_Xgb <- as.factor(clas_Xgb)             # riesgo quiera aceptar yo de que una persona
                                            # se suscribira a la cuenta de ahorros
clas_Xgb                                    # 1 significa si se sucribe
                                            # 0 significa que no se suscribe

data.test = data.frame(data.test, clas_Xgb) # lo añadimos al dataframe de Despliegue
data.test




statistics <- confusionMatrix(clas_Xgb,data.test$Adq_Ahorro,positive='1')
statistics     

table(clas_Xgb)                    # vemos lo predicho y lo real sobre el data set

table(data.test$Adq_Ahorro)

# Guardamos el modelo de ML
saveRDS(xgb_fit,"ModeloXG.rds")


#################################################################
##### 4 DESPLIEGUE DE ALGORITMOS DE MACHINE LEARNING ##########
#################################################################

# Leemos el dataset de despliegue de modelos #
DespliegueD<-read.csv("AdquisicionAhorroProspectos.csv",na.strings = c(""," ",NA)) 

# verificamos que no le falten datos

library(DataExplorer)
windows()
plot_missing(DespliegueD) # no hay datos missinsg asi que podemos continuar

# predecimos 

Adq_Ahorro=vector(mode='numeric', length=2281) # como el data set de despliegue no tiene
Adq_Ahorro                                     # no tiene la columna Adq_Ahorro, necesitamos
                                               # crearla

DespliegueD = data.frame(DespliegueD, Adq_Ahorro)  # efectivamente se añadio a la ultima
DespliegueD                                        # columna la etiqueta Adq_Ahorro

# cramos la matrix para el data set que vamos a predecir
MDes_XGB <- model.matrix(~ ., data=DespliegueD[,c(2:13)])
YDes <- as.vector(DespliegueD$Adq_Ahorro)
dDes <- xgb.DMatrix(MDes_XGB, label = YDes)

pred <- predict(xgb_fit, dDes, type="response")

# Convertimos la clase a probabilidad

clas_Xgb_des <- ifelse(pred<0.5,'0','1') # ponemos 0.5 pq es un estandar pero podriamos 
clas_Xgb_des<- as.factor(clas_Xgb_des)   # aumentar o disminuir este valor dependiendo del 
clas_Xgb_des                            # riesgo que queremos aceptar
table(clas_Xgb_des)

# guardamos los coddoc y el Adq_Ahorro predicho para el data set de Despliegue
library(forecast)
DespliegueD$Adq_Ahorro <- NULL #quitamos el que habiamos creado lleno de ceros
Adq_Ahorro=clas_Xgb_des # lo reemplazamos por el que acabamos de predecir   

DespliegueD = data.frame(DespliegueD, Adq_Ahorro) # lo añadimos al dataframe de Despliegue
DespliegueD

# creamos un data frame con solo el coddoc y La variable TARGET (Adq_Ahorro)
pred_xgboost <- data.frame(DespliegueD$coddoc,DespliegueD$Adq_Ahorro)

# Guardamos el data frame en formato ".csv"
write.csv(pred_xgboost,"Trabajo_Final.csv")

#################### Fin #############################################




