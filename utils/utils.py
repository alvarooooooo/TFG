import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import pandas as pd
import joblib


def get_instance_static_features(year, non_year_static_features):
    """
    :param year: año para calcular las características
    :return: devuelve las características estáticas para una instancia dado un año
    """
    # Años disponibles de CLC
    clc_all_years = [2006, 2012, 2018]

    # Calculo el año más cercano de CLC con respecto al año pasado por parametro
    year_differences = [abs(year - 2012) for year in clc_all_years]
    clc_year = clc_all_years[np.argmin(year_differences)]

    # Se crean las variables estáticas para un año concreto
    instance_static_features = non_year_static_features + [
        f"POP_DENS_{year}",
        f"CLC_{clc_year}",
        f"CLC_{clc_year}_0",
        f"CLC_{clc_year}_1",
        f"CLC_{clc_year}_2",
        f"CLC_{clc_year}_3",
        f"CLC_{clc_year}_4",
        f"CLC_{clc_year}_5",
        f"CLC_{clc_year}_6",
        f"CLC_{clc_year}_7",
        f"CLC_{clc_year}_8",
        f"CLC_{clc_year}_9",
    ]

    return instance_static_features


def create_instance(window, label, non_year_static_features, dynamic_features):
    """
    Crea una instancia junto con sus características y etiqueta dada una ventana de tiempo

    :param window: ventana de tiempo
    :param label: etiqueta de la instancia
    :return: devuelve la instancia junto con sus características,es decir, una fila de la tabla.

    """
    # Fila a añadir
    row_to_add = []

    # Para todas las características dinámicas realizamos la media
    for dynamic_var in dynamic_features:
        window_fire = 0.0
        # Si la característica es número de fuegos y es instancia de incendio
        if (dynamic_var == "number_of_fires") and (label):

            # Restamos el propio fuego del último día de la ventana
            window_fire = 0.1

        # Añado la media de la ventana
        row_to_add.append(window[dynamic_var].mean().item() - window_fire)

    # Idx del elemento que decide el año de la ventana
    idx_year = 4

    # Año de la ventana
    year = window["time"][idx_year].values.astype("datetime64[Y]").astype(np.datetime64)

    instance_static_features = get_instance_static_features(
        year, non_year_static_features
    )

    # Para las variables estáticas
    for static_var in instance_static_features:
        row_to_add.append(window[static_var].item())

    # Etiquetado
    row_to_add.append(label)

    return row_to_add


def plot_fwi(datacube, predictions, missing_idx):
    """
    Función para comparar modelo con el indice FWI. Muestra 4 graficos:
    Etiquetas reales de Burned + Ignition
    Puntos de Ignición según predicciones del modelo
    Indice FWI
    CLC

    :datacube: datacubo con informacion de Areas quemadas, puntos de ignición, indice FWI y CLC
    :predictions: vector de predicciones del modelo
    :missing_idx: indices no calculado por el modelo por ser agua o por contener valores Nan
    """
    # El array debe de tener tamaño 100
    if np.shape(predictions)[0] != 100:

        # Inserto valor no calculado por valores faltantes
        for idx in missing_idx:
            predictions = np.insert(predictions, idx, -1)

        # Los pixeles de agua no calculados se añaden con la etiqueta -1
        water_idx = np.argwhere(
            (
                (datacube["CLC_2012_9"] == 1) & (datacube["CLC_2012"] != 35)
            ).values.flatten()
        )
        for water_px in water_idx:
            predictions = np.insert(predictions, water_px, -1)

    # Transformo a matriz 10 x 10
    predictions = predictions.reshape(10, 10)

    red = np.array([1, 0, 0, 1])  # RGBA for red
    white = np.array([1, 1, 1, 1])  # RGBA for white

    # Defino colores
    ignition_color = ListedColormap([red])
    burned_color = ListedColormap([white])

    # Defino figura
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    #######################################################################################################
    # Burned areas + Corine Land Cover

    # Grafico el mapa con la variable estática Corine Land Cover 2012
    img = axes[0].imshow(datacube["CLC_2012"], cmap="viridis")

    plt.colorbar(
        img,
        ax=axes[0],
        label=f"{datacube['CLC_2012'].long_name} \n [{datacube['CLC_2012'].units}] ",
    )

    # Grafico los puntos quemados
    axes[0].imshow(
        np.ma.masked_where(
            datacube["burned_areas"] == 0,
            datacube["burned_areas"],
        ),
        cmap=burned_color,
    )

    # Grafico los puntos de ignición
    axes[0].imshow(
        np.ma.masked_where(
            datacube["ignition_points"] == 0, datacube["ignition_points"]
        ),
        cmap=ignition_color,
    )

    # Leyenda puntos de ignición
    ignition_areas_patch = Patch(color="red", label="Ignition Point")

    # Leyenda puntos quemados
    burned_areas_patch = Patch(color="white", label="Burned Areas")

    # Grafico leyenda
    axes[0].legend(
        handles=[burned_areas_patch, ignition_areas_patch], loc="upper right"
    )

    # Titulo
    date = datacube["time"].dt.strftime("%Y-%m-%dT%H:%M:%S").values
    axes[0].set_title(f"time={date}")

    # Elimino grid
    axes[0].grid(False)

    #######################################################################################################
    # Burned areas + Corine Land Cover + Ignition Points
    # Grafico el mapa con la variable estática Corine Land Cover 2012
    img = axes[1].imshow(datacube["CLC_2012"], cmap="viridis")

    plt.colorbar(
        img,
        ax=axes[1],
        label=f'{datacube["CLC_2012"].long_name} \n [{datacube["CLC_2012"].units}]',
    )

    # Grafico los puntos de ignición
    axes[1].imshow(
        np.ma.masked_where(predictions < 1, predictions),
        cmap=ignition_color,
    )

    # Leyenda puntos de ignición
    ignition_areas_patch = Patch(color="red", label="Ignition Point")

    # Grafico leyenda
    axes[1].legend(handles=[ignition_areas_patch], loc="upper right")

    # Titulo
    date = datacube["time"].dt.strftime("%Y-%m-%dT%H:%M:%S").values
    axes[1].set_title(f"time={date}")

    fig.suptitle("Burned Areas  VS  Burned Areas and Ignition Points", fontsize=16)

    # Elimino grid
    axes[1].grid(False)

    plt.tight_layout()
    plt.show()

    #############################################################################################################

    # Defino figura
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # FWI
    datacube["fwi"].plot.imshow(ax=axes[0])

    # Elimino grid
    axes[0].grid(False)

    #############################################################################################################
    # CLC

    datacube["CLC_2012"].plot.imshow(ax=axes[1])
    axes[1].grid(False)
    plt.tight_layout()
    plt.show()


def img_prep(img):
    """
    Función para preparar los datos de una venta o imagen para los algoritmos de ML

    :img: ventana
    """
    # Todos los indices
    all_idx = img.index

    # Elimino instancias con valores nan
    img = img.dropna()

    # Indices despues de limpieza
    cleaned_idx = img.index
    deleted_idx = all_idx.difference(cleaned_idx)

    img_data = img.drop(columns=["px", "py", "fwi", "label", "burned_areas"])
    img_labels = img.label

    # Clases que existen en test
    img_unique_clc = np.unique(img_data["CLC"])

    # Clases existentes en todo el dataset de Greecia
    all_clc_classes = np.arange(1, 45)
    all_clc_classes = np.append(all_clc_classes, 128)

    # Clases por añadir a One Hot encoding
    missing_classes = np.setdiff1d(all_clc_classes, img_unique_clc)
    missing_classes = [f"CLC_class_{clc_class}.0" for clc_class in missing_classes]

    # One Hot Encoding
    img_data_encoded = pd.get_dummies(
        img_data, prefix="CLC_class", columns=["CLC"], dtype=float
    )

    # Añado clases restantes
    img_data_encoded[missing_classes] = 0.0

    # Cargo el orden guardado
    ordered_one_hot_columns = joblib.load("./utils/ordered_one_hot_columns.pkl")

    # Orden de las columnas
    img_data_encoded = img_data_encoded.reindex(columns=ordered_one_hot_columns)

    # Cargo la definición
    scaler = joblib.load("./models/scaler.pkl")

    # Escalado de variables continuas
    img_data_encoded[img_data_encoded.columns[:-45]] = scaler.transform(
        img_data_encoded[img_data_encoded.columns[:-45]]
    )

    # Transformo en numpy array
    img_data_encoded = np.array(img_data_encoded)

    return img_data_encoded, img_labels, deleted_idx
