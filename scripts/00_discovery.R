################################################################################
# TÍTULO: nombre del script.R                                        #
# PROYECTO: Making MOney withg ML                                 #
# DESCRIPCIÓN: Descripción breve de lo que hace el script                                     #
# FECHA: escribir la fecha                      #
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
  skimr       # Resumen de datos
)

# Fijar semilla
set.seed(123)

###########################################
# 1. CARGAR Y EXPLORAR BASES DE DATOS    #
###########################################

# Cargar datos - train
train_properties <- read.csv("stores/raw/train.csv")

# Cargar datos - test
test_properties <- read.csv("stores/raw/test.csv")

# Mostrar dimensiones
cat("Dimensiones de train_properties:", dim(train_properties), "\n")
cat("Dimensiones de train_properties:", dim(test_properties), "\n")

# Exploración inicial
cat("\nEstructura de train_properties:\n")
str(train_properties)
cat("\nColumnas de train_proporties:\n")
print(colnames(train_properties))

cat("\nEstructura de test_properties:\n")
str(test_properties)
cat("\nColumnas de test_proporties:\n")
print(colnames(test_properties))


