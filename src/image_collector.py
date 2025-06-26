import requests
from PIL import Image
from io import BytesIO
from typing import Tuple, Optional

class GoogleMapsImageDownloader:
    """Download imagens aéreas usando Google Maps Static API."""
    
    def __init__(self, api_key: str):
        """
        Inicializa downloader.
        
        Args:
            api_key: Chave API do Google Maps
        """
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api/staticmap"
        
    def baixar_imagem_bbox(self, bbox: Tuple[float, float, float, float], 
                          size: Tuple[int, int] = (1024, 1024),
                          zoom: int = 18) -> Optional[Image.Image]:
        """
        Baixa imagem aérea para bbox especificado.
        
        Args:
            bbox: (min_lat, min_lon, max_lat, max_lon)
            size: (width, height) da imagem
            zoom: Nível de zoom (18 = ~0.6m/pixel)
            
        Returns:
            PIL Image ou None se erro
        """
        min_lat, min_lon, max_lat, max_lon = bbox
        
        # Calcular centro da bbox
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
        
        # Parâmetros da requisição
        params = {
            'center': f"{center_lat},{center_lon}",
            'zoom': zoom,
            'size': f"{size[0]}x{size[1]}",
            'maptype': 'satellite',
            'format': 'png',
            'key': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            
            if response.status_code == 200:
                # Converter bytes para PIL Image
                image = Image.open(BytesIO(response.content))
                return image
            else:
                print(f"Erro HTTP: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Erro ao baixar imagem: {e}")
            return None
    
    def calcular_zoom_para_bbox(self, bbox: Tuple[float, float, float, float], 
                               size: Tuple[int, int] = (1024, 1024)) -> int:
        """
        Calcula zoom ideal para cobrir toda a bbox.
        
        Args:
            bbox: (min_lat, min_lon, max_lat, max_lon)
            size: Tamanho da imagem
            
        Returns:
            Nível de zoom ideal
        """
        min_lat, min_lon, max_lat, max_lon = bbox
        
        # Diferença em graus
        lat_diff = max_lat - min_lat
        lon_diff = max_lon - min_lon
        
        # Calcular zoom baseado na diferença maior
        max_diff = max(lat_diff, lon_diff)
        
        # Aproximação: zoom 18 = ~0.004 graus por 1024 pixels
        # zoom 17 = ~0.008 graus, zoom 16 = ~0.016 graus, etc.
        zoom = 18
        coverage = 0.004
        
        while coverage < max_diff and zoom > 1:
            zoom -= 1
            coverage *= 2
            
        return max(1, min(20, zoom))