# funciones_auxiliares.py

from src.utilidades.database import comida_basedatos
from src.utilidades.constantes import GruposComida, DIAS_SEMANA, COMIDAS

comida_bd = comida_basedatos()

def calculo_macronutrientes(proteinas, carbohidratos, grasas):
    """Calcula el porcentaje de calorias provenientes de cada macronutriente."""

    calorias_proteinas = proteinas * 4
    calorias_carbohidratos = carbohidratos * 4
    calorias_grasas = grasas * 9

    total_calorias_macronutrientes = calorias_proteinas + calorias_carbohidratos + calorias_grasas

    porcentaje_proteinas = (calorias_proteinas / total_calorias_macronutrientes) * 100
    porcentaje_carbohidratos = (calorias_carbohidratos / total_calorias_macronutrientes) * 100
    porcentaje_grasas = (calorias_grasas / total_calorias_macronutrientes) * 100

    return porcentaje_proteinas, porcentaje_carbohidratos, porcentaje_grasas


def filtrar_comida(comida_bd, tipo, edad):
    """Filtra los alimentos segun el tipo de comida y la edad del usuario."""

    match tipo:
        case "almuerzo_cena":
            return [
                i for i, item in enumerate(comida_bd) if not item["grupo"].startswith(
                    (
                        GruposComida.Frutas.JUGOS_DE_FRUTAS[0],  # "FC"
                        GruposComida.Frutas.ZUMOS[0],  # "FE"
                        GruposComida.Bebidas.BEBIDAS[0],  # "P"
                        GruposComida.Alcohol.ALCOHOL[0],  # "Q"
                        GruposComida.Lacteos.LecheVaca.LECHE_VACA[0],  # "BA"
                        GruposComida.Lacteos.BEBIDAS_LACTEAS[0],  # "BH"
                        GruposComida.Bebidas.BebidasEnPolvoEsenciasInfusiones.BEBIDAS_EN_POLVO_ESENCIAS_INFUSIONES[0],  # "PA"
                        GruposComida.Azucares.AZUCARES[0],  # "S"
                        GruposComida.Cereales.CEREALES[0]  # "A"
                    )
                ) or item["grupo"] in {
                    GruposComida.Cereales.ARROZ[0],  # "AC"
                    GruposComida.Cereales.PASTA[0],  # "AD"
                    GruposComida.Cereales.PIZZAS[0],  # "AE"
                    GruposComida.Cereales.PANES[0]  # "AF"
                }
            ]
        
        case "bebidas":
            bebidas = [
                i for i, item in enumerate(comida_bd) if item["grupo"].startswith(
                    (
                        GruposComida.Bebidas.BEBIDAS[0],  # "P"
                        GruposComida.Frutas.JUGOS_DE_FRUTAS[0],  # "FC"
                        GruposComida.Frutas.ZUMOS[0]  # "FE"
                    )
                ) and not item["grupo"].startswith(
                    GruposComida.Bebidas.BebidasEnPolvoEsenciasInfusiones.BEBIDAS_EN_POLVO_ESENCIAS_INFUSIONES[0]  # "PA"
                )
            ]
            if edad >= 18:
                bebidas_alcoholicas = [
                    i for i, item in enumerate(comida_bd) if item["grupo"].startswith(
                        GruposComida.Alcohol.ALCOHOL[0]  # "Q"
                    )
                ]
                bebidas.extend(bebidas_alcoholicas)
            return bebidas
        
        case "desayuno":
            return [
                i for i, item in enumerate(comida_bd) if item["grupo"].startswith(
                (
                    GruposComida.Cereales.CEREALES[0],  # "A"
                    GruposComida.Huevos.HUEVOS[0],  # "C"
                    GruposComida.Frutas.FRUTAS_GENERALES[0],  # "FA"
                    GruposComida.Carne.CarneGeneral.BACON[0]  # "MAA"
                )
                ) and item["grupo"] not in {
                    GruposComida.Cereales.ARROZ[0],  # "AC"
                    GruposComida.Cereales.PASTA[0],  # "AD"
                    GruposComida.Cereales.PIZZAS[0]  # "AE"
                }
            ]
        
        case "bebida_desayuno":
            return [
                i for i, item in enumerate(comida_bd) if item["grupo"].startswith(
                    (
                        GruposComida.Lacteos.LecheVaca.LECHE_VACA[0],  # "BA"
                        GruposComida.Lacteos.BEBIDAS_LACTEAS[0],  # "BH"
                        GruposComida.Bebidas.BebidasEnPolvoEsenciasInfusiones.BEBIDAS_EN_POLVO_ESENCIAS_INFUSIONES[0],  # "PA"
                        GruposComida.Frutas.ZUMOS[0],  # "FE"
                        GruposComida.Frutas.JUGOS_DE_FRUTAS[0]  # "FC"
                    )
                )
            ]
        
        case "snacks":
            return [
                i for i, item in enumerate(comida_bd) if item["grupo"].startswith(
                    (
                        GruposComida.Frutas.FRUTAS[0],  # "F"
                        GruposComida.Azucares.AZUCARES[0]  # "S"
                    )
                )
            ]


def traducir_solucion(solucion, comida_bd):
    """Convierte la solucion de numeros en una lista de alimentos con sus nutrientes"""

    menu = {}
    datos_dia = {dia: {"calorias": 0, "proteinas": 0, "carbohidratos": 0, "grasas": 0} for dia in DIAS_SEMANA}
    
    indice = 0
    for dia in DIAS_SEMANA:
        menu[dia] = {}

        for comida in COMIDAS:
            num_alimentos = comida["num_alimentos"]
            alimentos = []
            calorias_totales = 0

            for _ in range(num_alimentos):

                # Traduce el indice a un alimento concreto
                if indice < len(solucion):
                    idx = int(solucion[indice])  
                    alimento = comida_bd[idx]
                    nombre_completo = f"- {alimento['nombre']} ({alimento['grupo']})"
                    alimentos.append(nombre_completo)

                    # Suma las calorias y macronutrientes del alimento
                    calorias_totales += alimento["calorias"]
                    datos_dia[dia]["calorias"] += alimento["calorias"]
                    datos_dia[dia]["proteinas"] += alimento["proteinas"]
                    datos_dia[dia]["carbohidratos"] += alimento["carbohidratos"]
                    datos_dia[dia]["grasas"] += alimento["grasas"]

                    indice += 1

            menu[dia][comida["nombre"]] = (alimentos, calorias_totales)
    
    # Calcula los porcentajes de macronutrientes para cada dia
    for dia in DIAS_SEMANA:
        calorias = datos_dia[dia]["calorias"]

        if calorias > 0:
            datos_dia[dia]["porcentaje_proteinas"], datos_dia[dia]["porcentaje_carbohidratos"], datos_dia[dia]["porcentaje_grasas"] = \
                calculo_macronutrientes(datos_dia[dia]["proteinas"], datos_dia[dia]["carbohidratos"], datos_dia[dia]["grasas"])
        else:
            datos_dia[dia]["porcentaje_proteinas"] = datos_dia[dia]["porcentaje_carbohidratos"] = datos_dia[dia]["porcentaje_grasas"] = 0

    return menu, datos_dia