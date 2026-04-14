import os
from dotenv import load_dotenv
import mysql.connector  # Librería MySQL

# Cargar variables de entorno
load_dotenv()

def conexion_basedatos():
    """Establece y devuelve la conexión a la base de datos."""
    config = {
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'host': os.getenv('DB_HOST'),
        'database': os.getenv('DB_NAME'),
        'raise_on_warnings': True
    }
    return mysql.connector.connect(**config)


def comida_basedatos():
    """Obtiene los datos de la tabla 'comida' y los devuelve como una lista de diccionarios."""

    cnx = conexion_basedatos()
    cursor = cnx.cursor(dictionary=True)

    query = "SELECT nombre, grupo, calorias, grasas, proteinas, carbohidratos FROM comida"
    cursor.execute(query)
    comida_basedatos = cursor.fetchall()

    cursor.close()
    cnx.close()
    
    return comida_basedatos


def sujetos_basedatos():
    """Obtiene los sujetos de la base de datos junto con sus gustos, disgustos y alergias."""

    cnx = conexion_basedatos()
    cursor = cnx.cursor(dictionary=True)

    # Obtener la información básica de los sujetos
    query_sujetos = """
    SELECT sp.id AS sujeto_id, sp.edad, sc.calorias 
    FROM sujetos sp
    JOIN sujetos_calorias sc ON sp.id = sc.id
    """
    cursor.execute(query_sujetos)
    sujetos = cursor.fetchall()

    # Obtener gustos, disgustos y alergias
    query_gustos = "SELECT sujeto_id, grupo FROM sujetos_gustos"
    query_disgustos = "SELECT sujeto_id, grupo FROM sujetos_disgustos"
    query_alergias = "SELECT sujeto_id, grupo FROM sujetos_alergias"

    cursor.execute(query_gustos)
    gustos = cursor.fetchall()

    cursor.execute(query_disgustos)
    disgustos = cursor.fetchall()

    cursor.execute(query_alergias)
    alergias = cursor.fetchall()

    # Procesar y estructurar la información
    sujetos_dict = {}
    for sujeto in sujetos:
        sujeto_id = sujeto["sujeto_id"]
        sujetos_dict[sujeto_id] = {
            "calorias": sujeto["calorias"],
            "edad": sujeto["edad"],
            "gustos": [],
            "disgustos": [],
            "alergias": []
        }

    # Asignar gustos a cada sujeto
    for gusto in gustos:
        sujetos_dict[gusto["sujeto_id"]]["gustos"].append(gusto["grupo"])

    # Asignar disgustos a cada sujeto
    for disgusto in disgustos:
        sujetos_dict[disgusto["sujeto_id"]]["disgustos"].append(disgusto["grupo"])

    # Asignar alergias a cada sujeto
    for alergia in alergias:
        sujetos_dict[alergia["sujeto_id"]]["alergias"].append(alergia["grupo"])

    cursor.close()
    cnx.close()

    return list(sujetos_dict.values())
