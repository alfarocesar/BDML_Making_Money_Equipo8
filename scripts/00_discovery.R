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

################################################################################
# ESTADÍSTICAS DESCRIPTIVAS GENERALES                                          #
################################################################################

cat("=============================================================\n")
cat("ESTADÍSTICAS DESCRIPTIVAS GENERALES PARA: Conjunto de Entrenamiento\n")
cat("=============================================================\n\n")

cat("Dimensiones: 38644 observaciones, 16 variables\n")
cat("Número de variables: 16\n")
cat("Número de observaciones: 38644\n\n")

cat("Tipos de datos:\n")
cat(" - character: 6\n - integer: 7\n - numeric: 2\n - logical: 1\n\n")

cat("Resumen de variables categóricas clave:\n")
cat(" - 'property_type' y 'operation_type' tienen solo 2 y 1 categoría respectivamente,\n   por lo que ofrecen poca variación.\n")
cat(" - 'city' es constante en todos los registros (Bogotá D.C.).\n")
cat(" - Las variables 'title' y 'description' presentan alto grado de unicidad y serán\n   útiles para generar nuevas variables de texto.\n\n")

cat("Resumen de variables numéricas clave:\n")
cat(" - 'price': valor promedio de 654 millones COP con desviación de 311 millones.\n")
cat(" - 'surface_total' y 'surface_covered' presentan más de 30,000 valores faltantes\n   (casi el 80% del total), por lo que no son confiables directamente.\n")
cat(" - 'rooms' tiene un 47% de valores faltantes, pero presenta fuerte correlación\n   con 'bedrooms', lo que sugiere redundancia.\n")
cat(" - 'bathrooms' tiene un 26% de valores faltantes, pero es informativa.\n")
cat(" - 'lat' y 'lon' están completas y serán claves para ingeniería espacial.\n")

cat("\n=============================================================\n")
cat("ESTADÍSTICAS DESCRIPTIVAS GENERALES PARA: Conjunto de Prueba\n")
cat("=============================================================\n\n")

cat("Dimensiones: 10286 observaciones, 16 variables\n")
cat("Número de variables: 16\n")
cat("Número de observaciones: 10286\n\n")

cat("Tipos de datos:\n")
cat(" - character: 6\n - integer: 7\n - numeric: 2\n - logical: 1\n\n")

cat("Observaciones clave:\n")
cat(" - Como se esperaba, la variable 'price' no está disponible en el conjunto de prueba.\n")
cat(" - La completitud de 'title' y 'description' es alta (superior al 99%), con miles de valores únicos,\n   lo que indica potencial para generar variables de texto mediante procesamiento de lenguaje natural (NLP).\n")
cat(" - Variables estructurales como 'surface_total' y 'surface_covered' también presentan\n   alta proporción de valores faltantes (>70%).\n")
cat(" - La completitud de 'bathrooms' es ligeramente mejor que en entrenamiento (76%), y 'rooms' tiene un patrón similar.\n")
cat(" - Las coordenadas geográficas ('lat', 'lon') están completas y coherentes.\n")

cat("\nEn resumen, los datos presentan buena calidad general con algunas limitaciones puntuales en variables\nestructurales. Esto requerirá imputación, creación de variables derivadas y selección de variables\nrobustas para los modelos predictivos.\n")

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


################################################################################
# ANÁLISIS DE VALORES FALTANTES                                                #
################################################################################

cat("==============================================================\n")
cat("ANÁLISIS DE VALORES FALTANTES PARA: Conjunto de Entrenamiento\n")
cat("==============================================================\n\n")

cat("Total de valores faltantes: 89,231\n")
cat("Porcentaje total de valores faltantes: 14.43%\n\n")

cat("Las variables con mayor proporción de ausencias son:\n")
cat(" - 'surface_total' (79.7%) y 'surface_covered' (77.8%), lo que compromete su uso directo\n   en modelos sin una estrategia robusta de imputación o eliminación.\n")
cat(" - 'rooms' presenta un 47.3% de datos faltantes. Sin embargo, su alto grado de correlación\n   con 'bedrooms' (disponible al 100%) sugiere que puede eliminarse sin pérdida de información relevante.\n")
cat(" - 'bathrooms' tiene un 26.1% de ausencias, pero será conservada dada su relevancia\n   para explicar el precio de los inmuebles.\n")
cat(" - 'title' y 'description' presentan muy bajo porcentaje de vacíos (<0.06%)\n   y pueden utilizarse tras una limpieza menor.\n")
cat(" - Variables clave como 'price', 'bedrooms', 'lat' y 'lon' están completamente disponibles.\n")

cat("Visualmente, se observa una clara concentración de faltantes en las variables estructurales.\n\n")

cat("==============================================================\n")
cat("ANÁLISIS DE VALORES FALTANTES PARA: Conjunto de Prueba\n")
cat("==============================================================\n\n")

cat("Total de valores faltantes: 33,248\n")
cat("Porcentaje total de valores faltantes: 20.2%\n\n")

cat("Observaciones clave:\n")
cat(" - La variable 'price' no está disponible en el conjunto de prueba, como es esperado,\n   ya que corresponde a la variable objetivo que se debe predecir.\n")
cat(" - 'surface_total' y 'surface_covered' presentan 81.9% y 72.5% de ausencias respectivamente,\n   lo que las hace aún menos confiables que en el conjunto de entrenamiento.\n")
cat(" - 'rooms' tiene un 44.5% de datos faltantes y 'bathrooms' un 24.2%, manteniendo patrones similares\n   al conjunto de entrenamiento.\n")
cat(" - Las variables 'title' y 'description' muestran tasas de completitud superiores al 99.9%,\n   lo cual es adecuado para tareas de ingeniería de texto.\n")
cat(" - Variables como 'lat', 'lon', 'bedrooms', 'year', y 'property_type' están completamente disponibles.\n")

cat("En conjunto, el tratamiento de valores faltantes deberá considerar estrategias diferenciadas:\n")
cat(" - Eliminación para variables con datos casi completamente ausentes ('surface_total', 'surface_covered').\n")
cat(" - Imputación (mediana, moda o modelos) para variables con faltantes intermedios ('bathrooms').\n")
cat(" - Descartar variables redundantes ('rooms') si existen correlatos completos ('bedrooms').\n")
cat(" - Conservar variables limpias para análisis espacial y textual.\n")
                         

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


################################################################################
# MÉTRICAS PARA CLASIFICACIÓN DE VARIABLES                                     #
################################################################################

cat("==============================================================\n")
cat("MÉTRICAS PARA CLASIFICACIÓN DE VARIABLES: Conjunto de Entrenamiento\n")
cat("==============================================================\n\n")

cat("El conjunto de entrenamiento cuenta con 16 variables, de las cuales 6 son de tipo 'character',\n7 'integer', 2 'numeric' y 1 'logical'. A continuación se destacan algunos hallazgos clave:\n\n")

cat("- Identificadores únicos:\n")
cat("  'property_id' tiene un 100% de unicidad y se utilizará únicamente como identificador.\n\n")

cat("- Variables categóricas:\n")
cat("  'property_type' contiene solo 2 categorías, y 'operation_type' y 'city' solo una, por lo que estas\n  últimas dos serán descartadas por no aportar variabilidad.\n")
cat("  'title' y 'description' tienen alta diversidad de valores (más de 16.000 y 30.000 únicos respectivamente),\n  lo que las convierte en candidatas ideales para extracción de características mediante NLP.\n\n")

cat("- Variables numéricas:\n")
cat("  'surface_total' y 'surface_covered' presentan alta variabilidad (435 y 368 valores únicos respectivamente),\n  pero también alta asimetría (skewness > 2.5) y curtosis, sugiriendo distribución sesgada con presencia de outliers.\n")
cat("  'bathrooms' tiene 13 valores distintos, y aunque su distribución también está sesgada (skewness = 1.42),\n  es relevante para caracterizar el inmueble.\n")
cat("  'price' muestra alta dispersión y sesgo (skewness = 1.16), con valores que van desde 30 a 1650 millones de COP,\n  lo cual justifica aplicar una transformación logarítmica para modelado.\n")
cat("  'lat' y 'lon' presentan alta granularidad y se usarán para crear variables espaciales.\n\n")

cat("==============================================================\n")
cat("MÉTRICAS PARA CLASIFICACIÓN DE VARIABLES: Conjunto de Prueba\n")
cat("==============================================================\n\n")

cat("El conjunto de prueba mantiene la misma estructura con 16 variables, pero sin la variable objetivo 'price'.\n\n")

cat("- Variables categóricas:\n")
cat("  'title' y 'description' siguen mostrando alto nivel de unicidad (más del 77% de los registros),\n  y se consideran apropiadas para técnicas de ingeniería de texto.\n")
cat("  'property_type' tiene 2 categorías, mientras que 'operation_type' y 'city' son constantes.\n\n")

cat("- Variables numéricas:\n")
cat("  'surface_total' y 'surface_covered' presentan distribuciones similares al entrenamiento, con gran dispersión\n  y valores extremos (kurtosis > 1000), lo que refuerza la necesidad de transformarlas o tratarlas con cautela.\n")
cat("  'bathrooms' y 'rooms' conservan patrones de variabilidad útiles, aunque con datos faltantes relevantes.\n")
cat("  'lat' y 'lon' permiten ubicar espacialmente las propiedades y tienen una cobertura completa.\n")

cat("\nResumen:\n")
cat("- Variables como 'price', 'surface_total', 'surface_covered' y 'rooms' requieren transformación o imputación.\n")
cat("- Variables con valores constantes serán descartadas ('city', 'operation_type').\n")
cat("- Se prioriza mantener 'bedrooms', 'bathrooms', 'lat', 'lon', y variables derivadas desde texto y OSM.\n")
                             

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

################################################################################
# CORRELACIONES ENTRE VARIABLES ESPECÍFICAS                                    #
################################################################################

cat("==============================================================\n")
cat("CORRELACIONES ENTRE VARIABLES ESPECÍFICAS: Conjunto de Entrenamiento\n")
cat("==============================================================\n\n")

cat("Se calcularon correlaciones bivariadas entre la variable 'bedrooms' y otras variables\nestructurales del inmueble con el objetivo de detectar redundancias y patrones consistentes.\n")

cat("Resultados clave:\n")
cat("- 'bedrooms' vs 'rooms': correlación de 0.9913, lo que sugiere una redundancia casi total.\n  Esto indica que mantener ambas variables en el modelo sería innecesario. Se recomienda conservar 'bedrooms',\n  dado que presenta menor proporción de valores faltantes.\n")
cat("- 'bedrooms' vs 'bathrooms': correlación moderada de 0.5870, lo cual sugiere que aunque están relacionadas,\n  aportan información distinta. Ambas pueden ser útiles en modelos predictivos.\n")
cat("- 'bedrooms' vs 'surface_covered': correlación positiva de 0.6635, lo que implica que una mayor cantidad de\n  habitaciones está asociada a una mayor área construida. Sin embargo, esta última variable presenta muchos valores\n  ausentes, por lo que su inclusión debe ser cuidadosamente evaluada.\n")
cat("- 'bedrooms' vs 'surface_total': correlación baja (0.2209), lo que sugiere escasa relación lineal directa.\n  Podría deberse a imprecisiones o alta dispersión en los datos de superficie total.\n")

cat("==============================================================\n")
cat("CORRELACIONES ENTRE VARIABLES ESPECÍFICAS: Conjunto de Prueba\n")
cat("==============================================================\n\n")

cat("Los patrones de correlación en el conjunto de prueba son en general coherentes con los observados\nen el conjunto de entrenamiento, aunque con leves diferencias por tamaño muestral y mayor cantidad de datos faltantes.\n")

cat("Resultados clave:\n")
cat("- 'bedrooms' vs 'rooms': correlación extremadamente alta de 0.9973, reafirmando la redundancia entre ambas.\n")
cat("- 'bedrooms' vs 'bathrooms': correlación de 0.6245, ligeramente superior a la del conjunto de entrenamiento.\n")
cat("- 'bedrooms' vs 'surface_covered': correlación de 0.7126, incluso mayor que en entrenamiento, lo que fortalece\n  la hipótesis de que esta variable captura características relevantes del inmueble.\n")
cat("- 'bedrooms' vs 'surface_total': correlación muy baja (0.0295), lo que refuerza su escaso aporte lineal.\n")

cat("\nConclusión:\n")
cat("La evidencia empírica justifica descartar 'rooms' por redundancia con 'bedrooms', y mantener 'bathrooms'\ncomo variable explicativa independiente. 'surface_covered' podría ser útil si se trata su alto porcentaje\n de valores faltantes, mientras que 'surface_total' no parece aportar valor predictivo adicional.\n")


#########################################################
# 7. DECISIONES SOBRE VARIABLES INICIALES               #
#########################################################

cat("
==============================================================
DECISIONES DE PREPROCESAMIENTO: VARIABLES INICIALES
==============================================================

A continuación, se documentan las decisiones tomadas para cada una de las variables originales, en función de su utilidad analítica, calidad de los datos, correlaciones observadas y relevancia para el objetivo del modelo:

- property_id (character): Se conservará como identificador único de cada observación. No será utilizada como predictor.
- city (character): Se eliminará dado que contiene un único valor ('Bogotá') en todas las observaciones, por lo que no aporta variación informativa.
- operation_type (character): Será eliminada por la misma razón; todas las observaciones corresponden a ventas.
- price (numeric): Esta es la variable objetivo (dependiente) del modelo de predicción. Dado su alto sesgo (skewness = 1.16), se recomienda aplicar una transformación logarítmica para mejorar la distribución y el ajuste del modelo.
- month, year (integer): Se conservarán y se sugiere combinarlas en una nueva variable de antigüedad relativa de la publicación (por ejemplo, 'meses_desde_2020').
- surface_total (integer): Se eliminará, ya que presenta >80% de valores faltantes en ambos conjuntos y una baja correlación con otras variables relevantes (correlación con 'bedrooms' ≈ 0.02–0.22).
- surface_covered (integer): Presenta entre 72% y 77% de datos faltantes, pero muestra una correlación relevante con 'bedrooms' (>0.66). Puede ser transformada e imputada condicionalmente si se decide incluirla.
- rooms (integer): Será eliminada debido a su redundancia con 'bedrooms' (correlación >0.99) y su 47% de datos faltantes en entrenamiento.
- bedrooms (integer): Se conservará como variable predictora principal de características estructurales. A pesar de su leve sesgo, no se transformará.
- bathrooms (integer): Se conservará. Aunque presenta entre 24% y 26% de faltantes, tiene una correlación útil con 'bedrooms' (>0.58) y aporta información relevante sobre el tamaño y funcionalidad del inmueble.
- lat, lon (numeric): Se conservarán. Serán transformadas en variables espaciales derivadas (por ejemplo, distancia a parques, densidad de servicios, etc.).
- property_type (character): Se conservará como variable categórica con bajo número de niveles. Se transformará mediante codificación dummy.
- title (character): A pesar de su alta unicidad, se utilizará para derivar nuevas variables mediante procesamiento de lenguaje natural (NLP), como número de palabras, presencia de ciertas palabras clave (‘terraza’, ‘estrato’, etc.).
- description (character): Se procesará para extraer variables textuales agregadas y dicotómicas relacionadas con características del inmueble y su entorno. Su cobertura es alta (>99.8%).

En resumen:
- Se eliminarán: 'surface_total', 'rooms', 'operation_type', 'city'.
- Se conservarán: 'price', 'month', 'year', 'bedrooms', 'bathrooms', 'lat', 'lon', 'property_type'.
- Se transformarán o derivarán: 'price' (log), 'title' y 'description' (NLP), 'month/year' (antigüedad), 'lat/lon' (distancias).
") 

################################################################################
#                            FIN DEL SCRIPT                                   #
################################################################################
