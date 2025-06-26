import requests
import json
import time
import random
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger('project')

class OSMDataCollector:
    """Coleta dados OSM para treinamento do modelo de detecção urbana."""
    
    def __init__(self, timeout: int = 30):
        """
        Inicializa coletor OSM.
        
        Args:
            timeout: Tempo limite para requests em segundos
        """
        self.timeout = timeout
        self.base_url = "https://overpass-api.de/api/interpreter"
        
    def extrair_dados_osm(self, bbox: Tuple[float, float, float, float]) -> Dict:
        """
        Extrai dados OSM para bbox especificado.
        
        Args:
            bbox: (min_lat, min_lon, max_lat, max_lon)
            
        Returns:
            Dict com features por categoria
        """
        min_lat, min_lon, max_lat, max_lon = bbox
        
        # Query Overpass para múltiplas categorias
        query = f"""
        [out:json][timeout:{self.timeout}];
        (
          way["building"]({min_lat},{min_lon},{max_lat},{max_lon});
          relation["building"]({min_lat},{min_lon},{max_lat},{max_lon});
          
          way["natural"="water"]({min_lat},{min_lon},{max_lat},{max_lon});
          way["waterway"]({min_lat},{min_lon},{max_lat},{max_lon});
          
          way["natural"="wood"]({min_lat},{min_lon},{max_lat},{max_lon});
          way["landuse"="forest"]({min_lat},{min_lon},{max_lat},{max_lon});
        );
        out geom;
        """
        
        try:
            response = requests.post(
                self.base_url, 
                data={'data': query},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return self._processar_resultado(result)
            else:
                logger.info(f"Erro HTTP: {response.status_code}")
                return self._estrutura_vazia()
                
        except Exception as e:
            logger.info(f"Erro na query OSM: {e}")
            return self._estrutura_vazia()
    
    def _processar_resultado(self, result: Dict) -> Dict:
        """Processa resultado da API Overpass."""
        dados = {
            'buildings': [],
            'water': [],
            'forest': [],
            'outros': []
        }
        
        elements = result.get('elements', [])
        
        for element in elements:
            categoria = self._categorizar_element(element)
            
            if categoria in dados:
                feature_data = {
                    'geometry': element.get('geometry', []),
                    'tags': element.get('tags', {}),
                    'id': element.get('id'),
                    'type': element.get('type')
                }
                dados[categoria].append(feature_data)
        
        return dados
    
    def _categorizar_element(self, element: Dict) -> str:
        """Determina categoria do element OSM."""
        tags = element.get('tags', {})
        
        # Buildings
        if 'building' in tags:
            return 'buildings'
            
        # Water
        if tags.get('natural') == 'water' or 'waterway' in tags:
            return 'water'
                
        # Forest/vegetation
        if tags.get('natural') == 'wood' or tags.get('landuse') == 'forest':
            return 'forest'
        
        return 'outros'
    
    def _estrutura_vazia(self) -> Dict:
        """Retorna estrutura vazia em caso de erro."""
        return {
            'buildings': [],
            'water': [],
            'forest': [],
            'outros': []
        }
    
    def obter_building_aleatorio(self, bbox_region: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """
        Obtém coordenadas de building aleatório.
        
        Args:
            bbox_region: Bbox da região para busca
            
        Returns:
            (latitude, longitude) do building
        """
        min_lat, min_lon, max_lat, max_lon = bbox_region
        
        query = f"""
        [out:json][timeout:{self.timeout}];
        (
          way["building"]({min_lat},{min_lon},{max_lat},{max_lon});
        );
        out center 10;
        """
        
        try:
            response = requests.post(
                self.base_url, 
                data={'data': query},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                elements = result.get('elements', [])
                
                if elements:
                    # Escolher building aleatório
                    element = random.choice(elements)
                    
                    if 'center' in element:
                        return (element['center']['lat'], element['center']['lon'])
                    elif 'lat' in element and 'lon' in element:
                        return (element['lat'], element['lon'])
                        
        except Exception as e:
            logger.info(f"Erro ao buscar building: {e}")
            
        # Fallback para coordenadas aleatórias na região
        return self.gerar_coordenadas_aleatorias(bbox_region)
    
    def gerar_coordenadas_aleatorias(self, region_bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """
        Gera coordenadas aleatórias dentro da região.
        
        Args:
            region_bbox: (min_lat, min_lon, max_lat, max_lon)
            
        Returns:
            (latitude, longitude) aleatórias
        """
        min_lat, min_lon, max_lat, max_lon = region_bbox
        
        lat = random.uniform(min_lat, max_lat)
        lon = random.uniform(min_lon, max_lon)
        
        return (lat, lon)
    
def criar_bbox_from_point(lat: float, lon: float, size_km: float = 0.5) -> Tuple[float, float, float, float]:
    """
    Cria bbox ao redor de ponto.
    
    Args:
        lat: Latitude central
        lon: Longitude central
        size_km: Tamanho do bbox em km
        
    Returns:
        (min_lat, min_lon, max_lat, max_lon)
    """
    # Aproximação: 1 grau ≈ 111 km
    delta = size_km / 111.0

    bbox = (
        lat - delta,
        lon - delta,
        lat + delta,
        lon + delta
    )

    logger.info(f"BBOX created: {bbox}")
    
    return bbox



def salvar_locations_json(locations: List[Dict], filename: str = "osm_locations.json"):
    """Salva locações em arquivo JSON."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(locations, f, indent=2, ensure_ascii=False)
    logger.info(f"Locações salvas em {filename}")


def carregar_locations_json(filename: str = "osm_locations.json") -> List[Dict]:
    """Carrega locações de arquivo JSON."""
    with open(filename, 'r', encoding='utf-8') as f:
        locations = json.load(f)
    logger.info(f"Carregadas {len(locations)} locações de {filename}")
    return locations


def testar_coleta_basica():
    """Testa coleta básica em ponto específico."""
    collector = OSMDataCollector()
    
    
    # Testar com Vilnius
    lat, lon = 54.6872, 25.2797
    bbox = criar_bbox_from_point(lat, lon, 0.5)
    

    logger.info(f"Testando coleta OSM para Vilnius")
    logger.info(f"Coordenadas: {lat}, {lon}")
    logger.info(f"Bbox: {bbox}")
    
    dados = collector.extrair_dados_osm(bbox)
    
    logger.info("Resultados:")
    for categoria, features in dados.items():
        logger.info(f"{categoria}\t: {len(features)}\t features")
        
    return dados


# Exemplo de uso
if __name__ == "__main__":
    # Teste básico primeiro
    logger.info("=== TESTE BÁSICO ===")
    testar_coleta_basica()
    
    
