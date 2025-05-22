################################################################################
# TÍTULO: 00_discovery.R                                                       #
# PROYECTO: Making Money with ML                                               #
# DESCRIPCIÓN: Análisis exploratorio inicial de los datos                      #
# FECHA: 20 de mayo de 2025                                                    #
################################################################################

# Configurar directorio de trabajo automáticamente
if (!require(rstudioapi)) install.packages("rstudioapi")
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
# Subir un nivel directorio para acceder a la estructura principal del proyecto
setwd("../")

# Cargar librerías
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  tidyverse,  # Manipulación de datos
  skimr,      # Resumen de datos
  scales,     # Formateo de escalas en gráficos
  moments,    # Cálculo de estadísticos como skewness y kurtosis
  ggplot2     # Visualización de datos
)

# Fijar semilla
set.seed(123)

###########################################
# 1. CARGAR BASES DE DATOS               #
###########################################

# Cargar datos - train
data_dir <- "stores/raw"
train_properties <- read.csv(file.path(data_dir, "train.csv"))

# Cargar datos - test
test_properties <- read.csv(file.path(data_dir, "test.csv"))

######################################################
# 2. EXPLORACIÓN INICIAL DE ESTRUCTURA DE LOS DATOS #
######################################################

cat("\n-------------------------------------------\n")
cat("EXPLORACIÓN INICIAL DE LAS BASES DE DATOS\n")
cat("-------------------------------------------\n")

cat("Dimensiones de train_properties:", dim(train_properties), "\n")
cat("Dimensiones de test_properties:", dim(test_properties), "\n")

cat("\nEstructura de train_properties:\n")
str(train_properties)
cat("\nColumnas de train_properties:\n")
print(colnames(train_properties))

cat("\nEstructura de test_properties:\n")
str(test_properties)
cat("\nColumnas de test_properties:\n")
print(colnames(test_properties))

###################################################
# 3. ESTADÍSTICAS DESCRIPTIVAS GENERALES         #
###################################################

# Función para crear un resumen general de los datos
generate_dataset_summary <- function(data, dataset_name) {
  cat(paste0("\n\n==============================================================\n"))
  cat(paste0("ESTADÍSTICAS DESCRIPTIVAS GENERALES PARA: ", dataset_name, "\n"))
  cat(paste0("==============================================================\n\n"))
  
  cat("Dimensiones:", dim(data), "\n")
  cat("Número de variables:", ncol(data), "\n")
  cat("Número de observaciones:", nrow(data), "\n\n")
  
  data_types <- sapply(data, class)
  cat("Tipos de datos:\n")
  print(table(data_types))
  cat("\n")
  
  cat("Estadísticas resumidas completas:\n")
  skim_result <- skim(data)
  print(skim_result)
  
  return(skim_result)
}

train_summary <- generate_dataset_summary(train_properties, "Conjunto de Entrenamiento")
test_summary <- generate_dataset_summary(test_properties, "Conjunto de Prueba")

###############################################
# 4. ANÁLISIS DE VALORES FALTANTES            #
###############################################

analyze_missing_values <- function(data, dataset_name) {
  cat(paste0("\n\n==============================================================\n"))
  cat(paste0("ANÁLISIS DE VALORES FALTANTES PARA: ", dataset_name, "\n"))
  cat(paste0("==============================================================\n\n"))
  
  missing_stats <- data.frame(
    variable = names(data),
    n_missing = sapply(data, function(x) sum(is.na(x))),
    pct_missing = sapply(data, function(x) mean(is.na(x)) * 100)
  ) %>% arrange(desc(pct_missing))
  
  cat("Resumen de valores faltantes por variable:\n")
  print(missing_stats)
  cat("\n")
  
  total_missing <- sum(is.na(data))
  total_cells <- nrow(data) * ncol(data)
  total_pct_missing <- (total_missing / total_cells) * 100
  
  cat("Total de valores faltantes:", total_missing, "\n")
  cat("Porcentaje total de valores faltantes:", round(total_pct_missing, 2), "%\n\n")
  
  missing_stats_filtered <- missing_stats %>% filter(n_missing > 0)
  
  if(nrow(missing_stats_filtered) > 0) {
    p <- ggplot(missing_stats_filtered, aes(x = reorder(variable, pct_missing), y = pct_missing)) +
      geom_bar(stat = "identity", fill = "steelblue") +
      coord_flip() +
      labs(title = paste("Porcentaje de valores faltantes -", dataset_name), x = "Variable", y = "Porcentaje de valores faltantes (%)") +
      theme_minimal() +
      scale_y_continuous(labels = function(x) paste0(x, "%"))
    print(p)
  } else {
    cat("No hay valores faltantes en el conjunto de datos.\n")
  }
  
  return(missing_stats)
}

train_missing <- analyze_missing_values(train_properties, "Conjunto de Entrenamiento")
test_missing <- analyze_missing_values(test_properties, "Conjunto de Prueba")

#####################################################
# 5. MÉTRICAS PARA CLASIFICACIÓN DE VARIABLES       #
#####################################################

analyze_variable_types <- function(data, dataset_name) {
  cat(paste0("\n\n==============================================================\n"))
  cat(paste0("MÉTRICAS PARA CLASIFICACIÓN DE VARIABLES: ", dataset_name, "\n"))
  cat(paste0("==============================================================\n\n"))
  
  var_metrics <- data.frame(
    variable = character(),
    data_type = character(),
    n_unique = numeric(),
    pct_unique = numeric(),
    is_integer_like = logical(),
    n_values = numeric(),
    min = numeric(),
    max = numeric(),
    mean = numeric(),
    median = numeric(),
    mode = character(),
    sd = numeric(),
    skewness = numeric(),
    kurtosis = numeric(),
    stringsAsFactors = FALSE
  )
  
  for (var in names(data)) {
    x <- data[[var]]
    data_type <- class(x)[1]
    n_unique <- length(unique(na.omit(x)))
    pct_unique <- n_unique / length(na.omit(x)) * 100
    n_values <- sum(!is.na(x))
    min_val <- max_val <- mean_val <- median_val <- sd_val <- skew_val <- kurt_val <- NA
    is_integer_like <- FALSE
    mode_val <- if (n_values > 0) {
      ux <- na.omit(x)
      ux_table <- table(ux)
      names(ux_table)[which.max(ux_table)]
    } else {
      NA
    }
    
    if (is.numeric(x)) {
      is_integer_like <- all(abs(x - round(x)) < 1e-10, na.rm = TRUE)
      min_val <- min(x, na.rm = TRUE)
      max_val <- max(x, na.rm = TRUE)
      mean_val <- mean(x, na.rm = TRUE)
      median_val <- median(x, na.rm = TRUE)
      sd_val <- sd(x, na.rm = TRUE)
      if (n_values > 3) {
        skew_val <- tryCatch(skewness(x, na.rm = TRUE), error = function(e) NA)
        kurt_val <- tryCatch(kurtosis(x, na.rm = TRUE), error = function(e) NA)
      }
    }
    
    var_metrics <- rbind(var_metrics, data.frame(
      variable = var,
      data_type = data_type,
      n_unique = n_unique,
      pct_unique = pct_unique,
      is_integer_like = is_integer_like,
      n_values = n_values,
      min = min_val,
      max = max_val,
      mean = mean_val,
      median = median_val,
      sd = sd_val,
      skewness = skew_val,
      kurtosis = kurt_val,
      mode = mode_val
    ))
  }
  
  var_metrics <- var_metrics %>% arrange(data_type, desc(n_unique))
  cat("Métricas para clasificación de variables:\n")
  print(var_metrics, row.names = FALSE)
  return(var_metrics)
}

train_var_metrics <- analyze_variable_types(train_properties, "Conjunto de Entrenamiento")
test_var_metrics <- analyze_variable_types(test_properties, "Conjunto de Prueba")

# Guardar resultados
output_dir <- "stores/processed"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(train_var_metrics, file.path(output_dir, "train_variable_metrics.csv"), row.names = FALSE)
write.csv(test_var_metrics, file.path(output_dir, "test_variable_metrics.csv"), row.names = FALSE)
write.csv(train_missing, file.path(output_dir, "train_missing_values.csv"), row.names = FALSE)
write.csv(test_missing, file.path(output_dir, "test_missing_values.csv"), row.names = FALSE)

###########################################
# 6. CORRELACIÓN ENTRE VARIABLES         #
###########################################

calculate_specific_correlations <- function(data, dataset_name) {
  cat(paste0("\n\n==============================================================\n"))
  cat(paste0("CORRELACIONES ENTRE VARIABLES ESPECÍFICAS: ", dataset_name, "\n"))
  cat(paste0("==============================================================\n\n"))
  
  var_pairs <- list(
    c("bedrooms", "surface_total"),
    c("bedrooms", "surface_covered"),
    c("bedrooms", "rooms"),
    c("bedrooms", "bathrooms")
  )
  
  corr_results <- data.frame(
    variable1 = character(),
    variable2 = character(),
    correlation = numeric(),
    n_observations = numeric(),
    stringsAsFactors = FALSE
  )
  
  for (pair in var_pairs) {
    var1 <- pair[1]; var2 <- pair[2]
    if (var1 %in% names(data) && var2 %in% names(data)) {
      complete_records <- data[!is.na(data[[var1]]) & !is.na(data[[var2]]), c(var1, var2)]
      n_obs <- nrow(complete_records)
      if (n_obs > 1) {
        corr_value <- cor(complete_records[[var1]], complete_records[[var2]])
        corr_results <- rbind(corr_results, data.frame(
          variable1 = var1,
          variable2 = var2,
          correlation = corr_value,
          n_observations = n_obs
        ))
      }
    }
  }
  
  if (nrow(corr_results) > 0) {
    cat("Correlaciones entre variables específicas:\n")
    print(corr_results, row.names = FALSE)
    p <- ggplot(corr_results, aes(x = paste(variable1, "vs", variable2), y = correlation)) +
      geom_bar(stat = "identity", fill = "steelblue") +
      coord_flip() +
      labs(title = paste("Correlaciones -", dataset_name), x = "Par de variables", y = "Coeficiente de correlación") +
      theme_minimal()
    print(p)
  } else {
    cat("No se pudo calcular ninguna correlación con los datos disponibles.\n")
  }
  
  return(corr_results)
}

train_correlations <- calculate_specific_correlations(train_properties, "Conjunto de Entrenamiento")
test_correlations <- calculate_specific_correlations(test_properties, "Conjunto de Prueba")

write.csv(train_correlations, file.path(output_dir, "train_specific_correlations.csv"), row.names = FALSE)
write.csv(test_correlations, file.path(output_dir, "test_specific_correlations.csv"), row.names = FALSE)

cat("\n\nAnálisis completado. Los resultados se han guardado en la carpeta 'stores/processed'.\n")

#########################################################
# 7. DECISIONES SOBRE VARIABLES INICIALES               #
#########################################################

cat("
==============================================================
DECISIONES DE PREPROCESAMIENTO: VARIABLES INICIALES
==============================================================

A continuación, se documentan las decisiones tomadas para cada una de las variables originales, en función de su utilidad analítica, calidad de los datos y relevancia para el objetivo del modelo:

- property_id (character): Se conservará como identificador único de cada observación. No será utilizada como predictor.
- city (character): Se eliminará dado que contiene un único valor ('Bogotá') en todas las observaciones, por lo que no aporta variación informativa.
- price (numeric): Esta es la variable objetivo (dependiente) del modelo de predicción y será conservada.
- month, year (integer): Se utilizarán para generar una nueva variable de antigüedad de la publicación (por ejemplo, en meses).
- surface_total (integer): Será eliminada, ya que presenta valores faltantes en el 100% de las observaciones del conjunto de prueba.
- surface_covered (integer): También se eliminará, dado que alrededor del 80% de sus valores están ausentes.
- rooms (integer): Se eliminará por dos razones: tiene un 47% de valores faltantes en el set de entrenamiento y presenta una correlación del 99% con 'bedrooms', por lo que no añade información adicional.
- bedrooms (integer): Se conservará como predictor clave. A pesar de su sesgo, no se aplicará transformación logarítmica y se mantendrá como variable numérica discreta.
- bathrooms (integer): Se conservará como variable predictora relevante de las características estructurales de la propiedad.
- property_type (character): Se conservará como variable categórica (por ejemplo, 'house' = 1, otros = 0).
- operation_type (character): Será eliminada, ya que todas las observaciones corresponden a operaciones de tipo 'venta', lo cual no aporta valor analítico.
- lat, lon (numeric): Se conservarán y utilizarán para crear variables espaciales derivadas como distancia a puntos de interés u otras métricas geográficas.
- title (character): No se utilizará directamente debido a que presenta aproximadamente un 24% de valores faltantes en el conjunto de prueba.
- description (character): Se utilizará para generar nuevas variables de texto a través de procesamiento de lenguaje natural (NLP).
")

################################################################################
#                            FIN DEL SCRIPT                                   #
################################################################################
