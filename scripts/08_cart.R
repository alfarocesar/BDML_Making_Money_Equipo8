################################################################################
# TÍTULO: 08_cart_model.R                                                      #
# PROYECTO: Making Money with ML                                               #
# DESCRIPCIÓN: Implementación de modelo CART para predicción de precios        #
# FECHA: 22 de mayo de 2025                                                    #
################################################################################

# Configurar directorio de trabajo automáticamente
if (!require(rstudioapi)) install.packages("rstudioapi")
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
# Subir un nivel directorio para acceder a la estructura principal del proyecto
setwd("../")

# Cargar librerías usando pacman
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  tidyverse,      # Manipulación de datos
  caret,          # Para entrenamiento de modelos
  rpart,          # Para árboles de decisión CART
  rpart.plot,     # Para visualizar árboles
  spatialsample,  # Muestreo espacial para modelos de aprendizaje automático
  sf              # Leer/escribir/manipular datos espaciales
)

# Fijar semilla para reproducibilidad
set.seed(123)

###########################################
# 1. CARGA Y PREPARACIÓN DE DATOS        #
###########################################

# Cargar datasets procesados
train <- read_csv("stores/processed/train_merged.csv")
test <- read_csv("stores/processed/test_merged.csv")

cat("Dimensiones train:", dim(train), "\n")
cat("Dimensiones test:", dim(test), "\n")

# Verificar que existe la variable precio
if(!"price" %in% names(train)) {
  stop("La variable 'price' no se encontró en el dataset de entrenamiento")
}

# Verificar que existen las coordenadas espaciales
if(!"lat" %in% names(train) || !"lon" %in% names(train)) {
  stop("Las variables 'lat' y/o 'lon' no se encontraron en el dataset de entrenamiento")
}

# Mostrar estadísticas básicas de la variable objetivo
cat("\nEstadísticas de price:\n")
summary(train$price)

###########################################
# 2. IMPUTACIÓN DE VALORES FALTANTES     #
###########################################

# Variables numéricas continuas - imputar con mediana
variables_numericas_continuas <- c("bedrooms", "antiguedad", "distancia_parque", 
                                   "distancia_universidad", "distancia_estacion_transporte", 
                                   "distancia_zona_comercial", "nivel_completitud")

for(var in variables_numericas_continuas) {
  if(var %in% names(train)) {
    if(sum(is.na(train[[var]])) > 0) {
      mediana <- median(train[[var]], na.rm = TRUE)
      train[[var]] <- ifelse(is.na(train[[var]]), mediana, train[[var]])
      test[[var]] <- ifelse(is.na(test[[var]]), mediana, test[[var]])
      cat("Imputada", var, "con mediana:", mediana, "\n")
    }
  }
}

# Variables ordinales discretas - imputar con moda
variables_ordinales <- c("nivel_premium", "nivel_venta_inmediata")

for(var in variables_ordinales) {
  if(var %in% names(train)) {
    if(sum(is.na(train[[var]])) > 0) {
      moda <- as.numeric(names(sort(table(train[[var]]), decreasing = TRUE))[1])
      train[[var]] <- ifelse(is.na(train[[var]]), moda, train[[var]])
      test[[var]] <- ifelse(is.na(test[[var]]), moda, test[[var]])
      cat("Imputada", var, "con moda:", moda, "\n")
    }
  }
}

# Variable binaria - imputar con moda
if("is_house" %in% names(train)) {
  if(sum(is.na(train$is_house)) > 0) {
    moda <- as.numeric(names(sort(table(train$is_house), decreasing = TRUE))[1])
    train$is_house <- ifelse(is.na(train$is_house), moda, train$is_house)
    test$is_house <- ifelse(is.na(test$is_house), moda, test$is_house)
    cat("Imputada is_house con moda:", moda, "\n")
  }
}

###########################################
# 3. ESPECIFICACIÓN DEL MODELO           #
###########################################

# Definir fórmula del modelo usando todas las variables predictoras disponibles
# (excluyendo property_id, price, lat y lon)
model_form <- price ~ bedrooms + antiguedad + is_house + 
  distancia_parque + distancia_universidad + distancia_estacion_transporte + 
  distancia_zona_comercial + nivel_premium + nivel_completitud + nivel_venta_inmediata

cat("\nFórmula del modelo:\n")
print(model_form)

###########################################
# 4. CONFIGURACIÓN DE VALIDACIÓN CRUZADA #
# ESPACIAL                                #
###########################################

# Convertir datos de entrenamiento a formato sf siguiendo el patrón de los cuadernos
train_sf <- st_as_sf(
  train,
  # "coords" is in x/y order -- so longitude goes first!
  coords = c("lon", "lat"),
  # Set our coordinate reference system to EPSG:4326,
  # the standard WGS84 geodetic coordinate reference system
  crs = 4326
)

cat("Datos convertidos a formato sf\n")
cat("Dimensiones train_sf:", dim(train_sf), "\n")

# Crear bloques espaciales siguiendo el patrón de los cuadernos
set.seed(123)
block_folds <- spatial_block_cv(train_sf, v = 5)

cat("Bloques espaciales creados:\n")
print(block_folds)

# Extraer índices de los folds espaciales para usar en caret
folds <- list()
for(i in 1:5) {
  folds[[i]] <- block_folds$splits[[i]]$in_id
}

# Configurar validación cruzada espacial
ctrl <- trainControl(
  method = "cv",        # Cross-validation
  number = 5,           # 5 folds
  index = folds,        # Usar índices de bloques espaciales
  verboseIter = TRUE    # Mostrar progreso
)

cat("Validación cruzada espacial configurada\n")

# Grilla de hiperparámetros para CART siguiendo el patrón de los cuadernos
grid <- expand.grid(cp = seq(0.001, 0.05, length.out = 20))

cat("Grilla de hiperparámetros cp:\n")
cat("Valores a probar:", nrow(grid), "\n")

###########################################
# 5. ENTRENAMIENTO DEL MODELO            #
###########################################

cat("Iniciando entrenamiento del modelo CART con validación cruzada espacial...\n")

# Entrenar modelo usando caret (siguiendo el patrón de los cuadernos)
set.seed(123)
modelo_cart <- train(
  model_form,           # Fórmula del modelo
  data = train,         # Datos de entrenamiento (formato original)
  method = 'rpart',     # CART (Classification And Regression Trees)
  trControl = ctrl,     # Configuración de CV espacial
  tuneGrid = grid       # Grilla de hiperparámetros
)

# Mostrar resultados del modelo
cat("Resultados del modelo:\n")
print(modelo_cart)

# Mostrar mejores parámetros
cat("Mejor valor de cp:", modelo_cart$bestTune$cp, "\n")

###########################################
# 6. VISUALIZACIÓN DEL ÁRBOL             #
###########################################

cat("Generando visualización del árbol...\n")

# Crear directorio plots si no existe
if (!dir.exists("views/plots")) {
  dir.create("views/plots", recursive = TRUE)
}

# Visualizar árbol con rpart.plot siguiendo el patrón de los cuadernos
png("views/plots/cart_tree.png", width = 12, height = 8, units = "in", res = 300)
rpart.plot::prp(modelo_cart$finalModel, 
                under = TRUE, 
                branch.lty = 2, 
                yesno = 2, 
                faclen = 0, 
                varlen = 15,
                box.palette = "Greens")
dev.off()

cat("Visualización del árbol guardada en: views/plots/cart_tree.png\n")

###########################################
# 7. PREDICCIONES                        #
###########################################

# Realizar predicciones en el conjunto de test
cat("Generando predicciones...\n")
predictions <- predict(modelo_cart, newdata = test)

# Verificar predicciones
cat("Estadísticas de las predicciones:\n")
cat("Min:", min(predictions, na.rm = TRUE), "\n")
cat("Max:", max(predictions, na.rm = TRUE), "\n")
cat("Mean:", mean(predictions, na.rm = TRUE), "\n")
cat("Median:", median(predictions, na.rm = TRUE), "\n")
cat("NAs:", sum(is.na(predictions)), "\n")

###########################################
# 8. EXPORTACIÓN DE RESULTADOS           #
###########################################

# Crear submission
submission <- data.frame(
  property_id = test$property_id,
  price = predictions
)

# Verificar submission
cat("Dimensiones del submission:", dim(submission), "\n")
cat("NAs en submission:", sum(is.na(submission$price)), "\n")

# Mostrar preview del submission
cat("Preview del submission:\n")
print(head(submission))

# Exportar submission
write_csv(submission, "stores/submissions/submission_C1.csv")

cat("Submission exportado exitosamente: stores/submissions/submission_3.csv\n")

###########################################
# 9. ANÁLISIS DE IMPORTANCIA DE VARIABLES#
###########################################

# Calcular importancia de variables
importance <- varImp(modelo_cart, scale = FALSE)
cat("Importancia de variables:\n")
print(importance)

###########################################
# 10. GUARDADO DE INFORMACIÓN DEL MODELO #
###########################################

# Crear directorio models si no existe
if (!dir.exists("stores/models")) {
  dir.create("stores/models", recursive = TRUE)
}

# Guardar información del modelo
model_info <- list(
  formula = model_form,
  best_tune = modelo_cart$bestTune,
  cv_results = modelo_cart$results,
  final_model = modelo_cart,
  variable_importance = importance,
  spatial_cv_used = TRUE,
  spatial_blocks = block_folds,
  date_created = Sys.time()
)

saveRDS(model_info, "stores/models/cart_model_info.rds")

cat("Información del modelo guardada en: stores/models/cart_model_info.rds\n")

###########################################
# 11. RESUMEN FINAL                      #
###########################################

cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("RESUMEN FINAL - CART MODEL CON VALIDACIÓN CRUZADA ESPACIAL\n")
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("Variables predictoras:\n")
cat("- Estructurales: bedrooms, antiguedad, is_house\n")
cat("- Espaciales: distancia_parque, distancia_universidad,\n")
cat("             distancia_estacion_transporte, distancia_zona_comercial\n")
cat("- De texto: nivel_premium, nivel_completitud, nivel_venta_inmediata\n")
cat("\nValidación cruzada: Espacial con bloques (5 folds)\n")
cat("Hiperparámetros óptimos:\n")
cat("Complexity Parameter (cp):", modelo_cart$bestTune$cp, "\n")
cat("\nMétricas de validación cruzada espacial:\n")
best_results <- modelo_cart$results[modelo_cart$results$cp == modelo_cart$bestTune$cp, ]
cat("RMSE:", round(best_results$RMSE, 2), "\n")
cat("R-squared:", round(best_results$Rsquared, 4), "\n")
cat("MAE:", round(best_results$MAE, 2), "\n")
cat("\nSubmission generado: submission_3.csv\n")
cat("Observaciones procesadas:", nrow(test), "\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

################################################################################
#                            FIN DEL SCRIPT                                   #
################################################################################