import os
from dotenv import load_dotenv

from config_logging import logger

from src.feature_collector import *
from src.image_collector import *
from menu import ask_user, MODE_QUESTION, BBOX_QUESTION, BBOX_DEFAULT_QUESTION, FILENAME_QUESTION

load_dotenv()

GOOGLE_MAPS_API_KEY = os.environ["GOOGLE_MAPS_API_KEY"]

def get_feature(lat : float, lon : float, size_km : float, api_key : str):
    pass

def get_image(file_name : str, lat = 54.6872, lon = 25.2797, size_km = 0.5):

    # Testar com Vilnius
    bbox = criar_bbox_from_point(lat, lon, size_km)

    downloader = GoogleMapsImageDownloader(GOOGLE_MAPS_API_KEY)
    imagem = downloader.baixar_imagem_bbox(bbox)

    if imagem:
        path = f"data/{file_name}.png"
        imagem.save(path)
        logger.info(f"Imagem salva em: {path}")

if __name__ == "__main__":
    mode = ask_user(MODE_QUESTION)['mode']


    match mode:
        case 'get_image':
            location = ask_user(BBOX_DEFAULT_QUESTION)['location']

            if location == 'vilnius':
                filename = ask_user(FILENAME_QUESTION)['filename']

                get_image(filename)


        case 'get_feature':
            bbox_answers = ask_user(BBOX_QUESTION)
            testar_coleta_basica()

    #     case

    # print(args)

    # collector = OSMDataCollector()

    # print(GOOGLE_MAPS_API_KEY)

    # # Testar com Vilnius
    # lat, lon = 54.6872, 25.2797
    # bbox = criar_bbox_from_point(lat, lon, 0.5)

    # downloader = GoogleMapsImageDownloader(GOOGLE_MAPS_API_KEY)
    # imagem = downloader.baixar_imagem_bbox(bbox)

    # if imagem:
    #     imagem.save("data/imagem_aerea.png")
    #     print("Imagem salva!")
    
    # print(f"Testando coleta OSM para Vilnius")
    # print(f"Coordenadas: {lat}, {lon}")
    # print(f"Bbox: {bbox}")
    
    # dados = collector.extrair_dados_osm(bbox)

    # print("\nResultados:")
    # for categoria, features in dados.items():
    #     print(f"  {categoria}: {len(features)} features")

    # print("=== TESTE BÁSICO ===")
    # testar_coleta_basica()

    # print("\n=== CRIANDO DADOS ===")
    # # Criar dataset de locações
    # locations = criar_dataset_osm_locations(
    #     num_buildings=10,  # Reduzido para teste
    #     num_random=25
    # )

    # print("\n=== RESULTADOS ===")
    # print(locations)
    
    # print("\n=== SALVAR RESULTADOS ===")
    # # Salvar resultados
    # salvar_locations_json(locations, "data/osm_dataset_locations.json")
    