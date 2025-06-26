import inquirer
import logging
import os

from inquirer.errors import ValidationError
from inquirer.themes import Default, BlueComposure
from typing import List, Any, Dict

logger = logging.getLogger('project')

def validar_sem_ponto(_, entrada_atual):
    """
    Valida se a entrada do usuário NÃO contém o caractere de ponto ('.').
    """
    # O operador 'in' verifica se o caractere '.' está dentro da string.
    if '.' in entrada_atual:
        # Se encontrar o ponto, levanta uma exceção com uma mensagem clara.
        raise ValidationError('', reason='O valor inserido não pode conter pontos!')
    
    # Se o 'if' for falso (não encontrou o ponto), a entrada é válida.
    return True

def limpar_tela():
    """Limpa o terminal de forma compatível com Windows, Linux e macOS."""
    # Para Windows
    if os.name == 'nt':
        _ = os.system('cls')
    # Para Linux e macOS (o nome é 'posix')
    else:
        _ = os.system('clear')

def validar_float(_, entrada_atual):
    """
    Valida se a entrada do usuário pode ser convertida para um float.
    Aceita tanto '.' quanto ',' como separador decimal.
    """
    # 1. Verifica se a entrada está vazia
    if not entrada_atual:
        raise ValidationError('', reason='O campo não pode ser vazio!')

    try:
        # 2. Substitui a vírgula por ponto para ser compatível com a função float()
        entrada_formatada = entrada_atual.replace(',', '.', 1)
        
        # 3. Tenta converter para float. Se conseguir, a validação passa.
        float(entrada_formatada)
        
        return True
    except ValueError:
        # 4. Se a conversão falhar, lança um erro para o usuário
        raise ValidationError('', reason='Formato inválido. Por favor, digite um número (ex: 10.50 ou 25,99).')

class MyTheme(Default):
    def __init__(self):
        super().__init__()
        self.Checkbox.selection_icon = "→"
        self.List.selection_cursor = "→"

MODE_QUESTION = [
  inquirer.List('mode',
                message="Oque você quer fazer?",
                choices=[
                    ("1. Get image from Google", 'get_image'),
                    ("2. Get features from Overpass API", 'get_feature'),
                    ],
            ),
]

BBOX_DEFAULT_QUESTION = [
    inquirer.List('location',
                message="Qual o local?",
                choices=[
                    ("1. Vilnius", 'vilnius'),
                    ("2. Quero escolher as coordenadas...", 'coordenadas'),
                    ],
            ),
]

BBOX_QUESTION = [
    inquirer.Text("lat", message="Latitude: ", validate=validar_float),
    inquirer.Text("long", message="Longitude: "),
    inquirer.Text("size_km", message="Size in kilometers: "),
]

FILENAME_QUESTION = [
    inquirer.Text("filename", message="Qual o nome do arquivo? ", validate=validar_sem_ponto),
]

def ask_user(questions : List[Any]) -> Dict[str, Any]:
    limpar_tela()
    answer = inquirer.prompt(questions, theme=MyTheme())
    return answer

if __name__ == '__main__':
    answer = ask_user(MODE_QUESTION)
    print(type(answer))