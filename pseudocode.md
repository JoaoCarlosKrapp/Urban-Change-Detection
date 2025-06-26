# Pseudocódigo: Sistema de Detecção de Mudanças Urbanas
## Baseado no artigo "Urban Change Detection from Aerial Images Using CNNs and Transfer Learning"

---

## 📋 CONFIGURAÇÕES INICIAIS

```pseudocode
DEFINIR CONSTANTES:
    CLASSES = ['casa', 'floresta', 'agua', 'outros']
    NUM_CLASSES = 4
    IMAGE_SIZE = 1024x1024
    RESOLUCAO_ALVO = 0.5  // metros por pixel
    
    // Normalização ImageNet
    MEAN_RGB = [0.485, 0.456, 0.406]
    STD_RGB = [0.229, 0.224, 0.225]
    
    // Hiperparâmetros de treinamento
    LEARNING_RATE_COARSE = 5e-4
    LEARNING_RATE_FINE = 5e-5
    MOMENTUM_COARSE = 0.5
    MOMENTUM_FINE = 0.1
    FOCAL_LOSS_ALPHA = 0.25
    FOCAL_LOSS_GAMMA = 2.0
```

---

## 🗂️ MÓDULO 1: PREPARAÇÃO DE DADOS

### 1.1 Coleta de Dados OSM
```pseudocode
FUNÇÃO extrair_dados_osm(coordenadas_bbox):
    PARA cada coordenada EM coordenadas_bbox:
        query_osm = CRIAR_QUERY([
            "way['building']",
            "way['natural'='water']", 
            "way['landuse'='forest']",
            "way['natural'='wood']"
        ])
        
        dados_vetoriais = EXECUTAR_QUERY_OVERPASS(query_osm)
        
        RETORNAR dados_vetoriais
```

### 1.2 Normalização de Imagens
```pseudocode
FUNÇÃO normalizar_imagem(imagem_path):
    // Carregar imagem
    imagem = CARREGAR_IMAGEM(imagem_path)
    
    // 1. Redimensionar para resolução alvo
    imagem_redimensionada = REDIMENSIONAR(imagem, IMAGE_SIZE)
    
    // 2. Normalização de contraste (percentis 2-98%)
    PARA cada canal EM [R, G, B]:
        p2, p98 = CALCULAR_PERCENTIS(imagem_redimensionada[canal], [2, 98])
        imagem_redimensionada[canal] = CLIP(imagem_redimensionada[canal], p2, p98)
    
    // 3. Normalização ImageNet
    imagem_float = imagem_redimensionada / 255.0
    PARA i EM [0, 1, 2]:
        imagem_float[i] = (imagem_float[i] - MEAN_RGB[i]) / STD_RGB[i]
    
    RETORNAR imagem_float
```

### 1.3 Criação de Labels (Rasterização)
```pseudocode
FUNÇÃO criar_labels(dados_osm, coordenadas_imagem):
    label_mask = CRIAR_MATRIZ_ZEROS(IMAGE_SIZE, IMAGE_SIZE)
    
    PARA cada feature EM dados_osm:
        SE feature.tipo == 'building':
            classe_id = 0  // casa
        SENÃO SE feature.tipo == 'water':
            classe_id = 2  // agua
        SENÃO SE feature.tipo EM ['forest', 'wood']:
            classe_id = 1  // floresta
        SENÃO:
            classe_id = 3  // outros
        
        // Rasterizar polígono para pixels
        pixels_polygon = RASTERIZAR_POLIGONO(feature.geometry, coordenadas_imagem)
        label_mask[pixels_polygon] = classe_id
    
    RETORNAR label_mask
```

### 1.4 Geração de Dataset Coarse
```pseudocode
FUNÇÃO gerar_dataset_coarse(num_amostras=5000):
    dataset_coarse = []
    
    // 4000 imagens centradas em buildings
    PARA i EM range(4000):
        building_random = SELECIONAR_BUILDING_ALEATORIO_OSM()
        coordenadas = OBTER_COORDENADAS(building_random)
        
        PARA periodo EM ['2009-2010', '2012-2013', '2015-2017']:
            imagem = BAIXAR_IMAGEM_AEREA(coordenadas, periodo)
            imagem_norm = normalizar_imagem(imagem)
            
            dados_osm = extrair_dados_osm(coordenadas)
            labels = criar_labels(dados_osm, coordenadas)
            
            dataset_coarse.ADICIONAR({
                'imagem': imagem_norm,
                'labels': labels,
                'periodo': periodo,
                'coordenadas': coordenadas
            })
    
    // 1000 imagens com pontos aleatórios (vegetação)
    PARA i EM range(1000):
        ponto_aleatorio = GERAR_COORDENADAS_ALEATORIAS()
        
        PARA periodo EM ['2009-2010', '2012-2013', '2015-2017']:
            imagem = BAIXAR_IMAGEM_AEREA(ponto_aleatorio, periodo)
            imagem_norm = normalizar_imagem(imagem)
            
            dados_osm = extrair_dados_osm(ponto_aleatorio)
            labels = criar_labels(dados_osm, ponto_aleatorio)
            
            dataset_coarse.ADICIONAR({
                'imagem': imagem_norm,
                'labels': labels,
                'periodo': periodo,
                'coordenadas': ponto_aleatorio
            })
    
    RETORNAR dataset_coarse
```

### 1.5 Criação de Dataset Fine-tuning
```pseudocode
FUNÇÃO gerar_dataset_fine_tuning(num_amostras=321):
    dataset_fine = []
    dataset_candidatos = gerar_dataset_coarse(num_amostras)
    
    PARA cada amostra EM dataset_candidatos:
        // Revisão manual
        EXIBIR_IMAGEM_E_LABELS(amostra.imagem, amostra.labels)
        aprovacao = SOLICITAR_APROVACAO_MANUAL()
        
        SE aprovacao == VERDADEIRO:
            // Permitir correção manual se necessário
            labels_corrigidos = PERMITIR_CORRECAO_MANUAL(amostra.labels)
            
            dataset_fine.ADICIONAR({
                'imagem': amostra.imagem,
                'labels': labels_corrigidos,
                'periodo': amostra.periodo,
                'coordenadas': amostra.coordenadas
            })
    
    RETORNAR dataset_fine
```

---

## 🧠 MÓDULO 2: ARQUITETURA DO MODELO

### 2.1 Definição do Modelo
```pseudocode
FUNÇÃO criar_modelo_deeplabv3():
    // Inicializar DeepLabv3 com backbone ResNet50
    modelo = DeepLabv3(
        backbone='ResNet50',
        num_classes=NUM_CLASSES,
        output_stride=16,
        pretrained_backbone=VERDADEIRO
    )
    
    // Carregar pesos pré-treinados do ImageNet
    modelo.CARREGAR_PESOS_IMAGENET()
    
    RETORNAR modelo
```

### 2.2 Função de Loss (Focal Loss)
```pseudocode
FUNÇÃO focal_loss(predicoes, targets):
    // Implementação da Focal Loss
    PARA cada pixel EM predicoes:
        p = SOFTMAX(predicoes[pixel])
        target_class = targets[pixel]
        
        pt = p[target_class]
        
        // Focal Loss: FL(pt) = -α(1-pt)^γ log(pt)
        loss_pixel = -FOCAL_LOSS_ALPHA * POW(1 - pt, FOCAL_LOSS_GAMMA) * LOG(pt)
    
    loss_total = MEDIA(loss_pixel)
    RETORNAR loss_total
```

---

## 🏋️ MÓDULO 3: PROCESSO DE TREINAMENTO

### 3.1 Passo 1: Modelo Base (M1)
```pseudocode
FUNÇÃO passo_1_modelo_base():
    // M1: Apenas pesos pré-treinados do ImageNet
    modelo_M1 = criar_modelo_deeplabv3()
    
    IMPRIMIR("Passo 1: Modelo M1 carregado com pesos ImageNet")
    RETORNAR modelo_M1
```

### 3.2 Passo 2: Treinamento Coarse (M12)
```pseudocode
FUNÇÃO passo_2_treinamento_coarse(modelo_M1, dataset_coarse):
    modelo_M12 = COPIAR_MODELO(modelo_M1)
    
    // Configurar otimizador
    otimizador = SGDM(
        parametros=modelo_M12.parametros,
        learning_rate=LEARNING_RATE_COARSE,
        momentum=MOMENTUM_COARSE
    )
    
    // Divisão train/validation
    train_coarse, val_coarse = DIVIDIR_DATASET(dataset_coarse, 0.8)
    
    PARA epoca EM range(50):  // 50 épocas como no artigo
        modelo_M12.TREINAR()
        
        loss_epoca = 0
        PARA batch EM train_coarse:
            // Forward pass
            predicoes = modelo_M12(batch.imagens)
            loss = focal_loss(predicoes, batch.labels)
            
            // Backward pass
            otimizador.ZERAR_GRADIENTES()
            loss.BACKWARD()
            otimizador.STEP()
            
            loss_epoca += loss
        
        // Validação
        SE epoca % 10 == 0:
            mIoU_val = AVALIAR_MODELO(modelo_M12, val_coarse)
            IMPRIMIR(f"Época {epoca}: Loss={loss_epoca}, mIoU={mIoU_val}")
    
    RETORNAR modelo_M12
```

### 3.3 Passo 3: Fine-tuning (M123)
```pseudocode
FUNÇÃO passo_3_fine_tuning(modelo_M12, dataset_fine):
    modelo_M123 = COPIAR_MODELO(modelo_M12)
    
    // Configurar otimizador para fine-tuning
    otimizador = SGDM(
        parametros=modelo_M123.parametros,
        learning_rate=LEARNING_RATE_FINE,  // Learning rate menor
        momentum=MOMENTUM_FINE
    )
    
    // Divisão train/validation
    train_fine, val_fine = DIVIDIR_DATASET(dataset_fine, 0.9)
    
    PARA epoca EM range(100):  // 100 épocas como no artigo
        modelo_M123.TREINAR()
        
        loss_epoca = 0
        PARA batch EM train_fine:
            predicoes = modelo_M123(batch.imagens)
            loss = focal_loss(predicoes, batch.labels)
            
            otimizador.ZERAR_GRADIENTES()
            loss.BACKWARD()
            otimizador.STEP()
            
            loss_epoca += loss
        
        // Validação a cada 10 épocas
        SE epoca % 10 == 0:
            mIoU_val = AVALIAR_MODELO(modelo_M123, val_fine)
            pixel_accuracy = CALCULAR_PIXEL_ACCURACY(modelo_M123, val_fine)
            IMPRIMIR(f"Época {epoca}: Loss={loss_epoca}, mIoU={mIoU_val}, PixelAcc={pixel_accuracy}")
    
    RETORNAR modelo_M123
```

### 3.4 Função de Avaliação
```pseudocode
FUNÇÃO AVALIAR_MODELO(modelo, dataset_validacao):
    modelo.AVALIAR()
    
    total_iou = 0
    total_pixel_accuracy = 0
    confusion_matrix = CRIAR_MATRIZ_ZEROS(NUM_CLASSES, NUM_CLASSES)
    
    PARA amostra EM dataset_validacao:
        predicao = modelo(amostra.imagem)
        classe_predita = ARGMAX(predicao, dim=1)
        
        // Calcular IoU para cada classe
        PARA classe EM range(NUM_CLASSES):
            intersecao = SOMA((classe_predita == classe) AND (amostra.labels == classe))
            uniao = SOMA((classe_predita == classe) OR (amostra.labels == classe))
            iou_classe = intersecao / uniao SE uniao > 0 SENÃO 0
            total_iou += iou_classe
        
        // Calcular pixel accuracy
        pixels_corretos = SOMA(classe_predita == amostra.labels)
        total_pixels = classe_predita.tamanho
        pixel_acc = pixels_corretos / total_pixels
        total_pixel_accuracy += pixel_acc
        
        // Atualizar matriz de confusão
        PARA i EM range(IMAGE_SIZE):
            PARA j EM range(IMAGE_SIZE):
                real = amostra.labels[i][j]
                pred = classe_predita[i][j]
                confusion_matrix[real][pred] += 1
    
    mIoU = total_iou / (len(dataset_validacao) * NUM_CLASSES)
    pixel_accuracy_media = total_pixel_accuracy / len(dataset_validacao)
    
    RETORNAR mIoU, pixel_accuracy_media, confusion_matrix
```

---

## 🎯 MÓDULO 4: APLICAÇÃO E INFERÊNCIA

### 4.1 Inferência em Nova Imagem
```pseudocode
FUNÇÃO inferir_mudancas(modelo_treinado, imagem_path):
    // Carregar e normalizar imagem
    imagem = normalizar_imagem(imagem_path)
    
    // Inferência
    modelo_treinado.AVALIAR()
    COM torch.no_grad():
        predicao = modelo_treinado(imagem.unsqueeze(0))
        classe_predita = ARGMAX(predicao, dim=1)
        probabilidades = SOFTMAX(predicao, dim=1)
    
    RETORNAR classe_predita, probabilidades
```

### 4.2 Detecção de Mudanças Entre Períodos
```pseudocode
FUNÇÃO detectar_mudancas(modelo, imagem1_path, imagem2_path):
    // Inferir classes para ambas as imagens
    classes1, prob1 = inferir_mudancas(modelo, imagem1_path)
    classes2, prob2 = inferir_mudancas(modelo, imagem2_path)
    
    // Calcular mapa de mudanças
    mapa_mudancas = CRIAR_MATRIZ_ZEROS(IMAGE_SIZE, IMAGE_SIZE, 3)  // RGB
    
    PARA i EM range(IMAGE_SIZE):
        PARA j EM range(IMAGE_SIZE):
            SE classes1[i][j] != classes2[i][j]:
                // Mudança detectada
                classe_antiga = classes1[i][j]
                classe_nova = classes2[i][j]
                
                // Codificar mudança com cores
                SE classe_antiga == 0 AND classe_nova == 3:  // casa -> outros
                    mapa_mudancas[i][j] = [255, 0, 0]  // vermelho (demolição)
                SENÃO SE classe_antiga == 3 AND classe_nova == 0:  // outros -> casa
                    mapa_mudancas[i][j] = [0, 255, 0]  // verde (construção)
                SENÃO SE classe_antiga == 1 AND classe_nova != 1:  // floresta removida
                    mapa_mudancas[i][j] = [255, 255, 0]  // amarelo (desmatamento)
                SENÃO:
                    mapa_mudancas[i][j] = [0, 0, 255]  // azul (outras mudanças)
            SENÃO:
                // Sem mudança
                mapa_mudancas[i][j] = [255, 255, 255]  // branco
    
    RETORNAR mapa_mudancas
```

### 4.3 Análise em Múltiplos Níveis
```pseudocode
FUNÇÃO analise_multiescala(modelo, lista_imagens, coordenadas):
    resultados = {}
    
    // NÍVEL 1: Análise Local (Pixel-level)
    resultados['local'] = []
    PARA imagem EM lista_imagens:
        segmentacao = inferir_mudancas(modelo, imagem)
        resultados['local'].ADICIONAR(segmentacao)
    
    // NÍVEL 2: Análise de Grid (Cell-level)
    grid_size = 64  // Dividir imagem em células 64x64
    resultados['grid'] = CRIAR_MATRIZ_ZEROS(IMAGE_SIZE//grid_size, IMAGE_SIZE//grid_size)
    
    PARA cada célula EM grade:
        contagem_buildings = CONTAR_PIXELS_CLASSE(resultados['local'], classe=0, celula)
        resultados['grid'][celula] = contagem_buildings
    
    // NÍVEL 3: Análise Municipal (Region-level)
    resultados['municipal'] = {
        'total_buildings': SOMA(resultados['grid']),
        'densidade_urbana': resultados['municipal']['total_buildings'] / AREA_TOTAL,
        'mudanca_percentual': CALCULAR_MUDANCA_TEMPORAL(lista_imagens)
    }
    
    RETORNAR resultados
```

---

## 🚀 MÓDULO 5: PIPELINE PRINCIPAL

### 5.1 Função Principal de Treinamento
```pseudocode
FUNÇÃO main_treinamento():
    IMPRIMIR("=== INICIANDO PIPELINE DE TREINAMENTO ===")
    
    // 1. Preparar dados
    IMPRIMIR("1. Gerando dataset coarse...")
    dataset_coarse = gerar_dataset_coarse(5000)
    
    IMPRIMIR("2. Gerando dataset fine-tuning...")
    dataset_fine = gerar_dataset_fine_tuning(321)
    
    // 2. Treinar modelo em 3 etapas
    IMPRIMIR("3. Passo 1: Carregando modelo base...")
    modelo_M1 = passo_1_modelo_base()
    
    IMPRIMIR("4. Passo 2: Treinamento coarse...")
    modelo_M12 = passo_2_treinamento_coarse(modelo_M1, dataset_coarse)
    
    IMPRIMIR("5. Passo 3: Fine-tuning...")
    modelo_M123 = passo_3_fine_tuning(modelo_M12, dataset_fine)
    
    // 3. Avaliar modelo final
    IMPRIMIR("6. Avaliação final...")
    mIoU_final, pixel_acc, conf_matrix = AVALIAR_MODELO(modelo_M123, dataset_fine)
    
    IMPRIMIR(f"Resultados finais:")
    IMPRIMIR(f"  - mIoU: {mIoU_final:.4f}")
    IMPRIMIR(f"  - Pixel Accuracy: {pixel_acc:.4f}")
    
    // 4. Salvar modelo
    SALVAR_MODELO(modelo_M123, "modelo_urban_change_M123.pth")
    
    RETORNAR modelo_M123
```

### 5.2 Função Principal de Aplicação
```pseudocode
FUNÇÃO main_aplicacao(modelo_path, imagens_entrada):
    IMPRIMIR("=== APLICANDO MODELO TREINADO ===")
    
    // Carregar modelo treinado
    modelo = CARREGAR_MODELO(modelo_path)
    
    resultados_finais = {}
    
    PARA i EM range(len(imagens_entrada)):
        IMPRIMIR(f"Processando imagem {i+1}/{len(imagens_entrada)}")
        
        // Inferência individual
        segmentacao = inferir_mudancas(modelo, imagens_entrada[i])
        
        // Detecção de mudanças (se há mais de uma imagem)
        SE len(imagens_entrada) > 1 AND i > 0:
            mudancas = detectar_mudancas(modelo, 
                                       imagens_entrada[i-1], 
                                       imagens_entrada[i])
            resultados_finais[f'mudancas_{i-1}_para_{i}'] = mudancas
        
        resultados_finais[f'segmentacao_{i}'] = segmentacao
    
    // Análise multiescala
    SE len(imagens_entrada) >= 2:
        analise_completa = analise_multiescala(modelo, imagens_entrada, None)
        resultados_finais['analise_multiescala'] = analise_completa
    
    RETORNAR resultados_finais
```

---

## 📊 MÓDULO 6: MÉTRICAS E VISUALIZAÇÃO

### 6.1 Calcular Métricas Detalhadas
```pseudocode
FUNÇÃO calcular_metricas_detalhadas(modelo, dataset_teste):
    metricas = {}
    
    // Para cada classe
    PARA classe EM range(NUM_CLASSES):
        nome_classe = CLASSES[classe]
        
        precision = CALCULAR_PRECISION(modelo, dataset_teste, classe)
        recall = CALCULAR_RECALL(modelo, dataset_teste, classe)
        f1_score = 2 * (precision * recall) / (precision + recall)
        iou = CALCULAR_IOU(modelo, dataset_teste, classe)
        
        metricas[nome_classe] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'iou': iou
        }
    
    RETORNAR metricas
```

### 6.2 Visualização de Resultados
```pseudocode
FUNÇÃO visualizar_resultados(imagem_original, segmentacao, mapa_mudancas=None):
    // Criar visualização colorida
    cores_classes = {
        0: [255, 0, 255],    // casa - magenta
        1: [139, 69, 19],    // floresta - marrom
        2: [0, 0, 255],      // água - azul
        3: [255, 255, 255]   // outros - branco
    }
    
    imagem_colorida = CRIAR_MATRIZ_ZEROS(IMAGE_SIZE, IMAGE_SIZE, 3)
    
    PARA i EM range(IMAGE_SIZE):
        PARA j EM range(IMAGE_SIZE):
            classe = segmentacao[i][j]
            imagem_colorida[i][j] = cores_classes[classe]
    
    // Exibir resultados
    SUBPLOT(1, 3, 1)
    MOSTRAR_IMAGEM(imagem_original, titulo="Imagem Original")
    
    SUBPLOT(1, 3, 2)
    MOSTRAR_IMAGEM(imagem_colorida, titulo="Segmentação")
    
    SE mapa_mudancas NÃO É None:
        SUBPLOT(1, 3, 3)
        MOSTRAR_IMAGEM(mapa_mudancas, titulo="Mapa de Mudanças")
    
    MOSTRAR_PLOTS()
```

---

## 🔧 EXECUÇÃO DO SISTEMA

```pseudocode
// EXECUÇÃO PRINCIPAL
SE modo == "TREINAMENTO":
    modelo_treinado = main_treinamento()
    
SENÃO SE modo == "APLICACAO":
    imagens = ["imagem1.tif", "imagem2.tif", "imagem3.tif"]
    resultados = main_aplicacao("modelo_urban_change_M123.pth", imagens)
    
    // Visualizar resultados
    PARA chave, valor EM resultados.items():
        SE "segmentacao" EM chave:
            visualizar_resultados(imagens[0], valor)
        SENÃO SE "mudancas" EM chave:
            visualizar_resultados(None, None, valor)

IMPRIMIR("=== PIPELINE CONCLUÍDO ===")
```

---

## 📝 NOTAS DE IMPLEMENTAÇÃO

### Dependências Principais:
- Framework: PyTorch/MXNet + GluonCV
- Geoespaciais: GDAL, Rasterio, OSMnx
- Processamento: OpenCV, NumPy, Pandas
- Visualização: Matplotlib, QGIS (para análise avançada)

### Considerações de Performance:
- Usar GPU para treinamento (NVIDIA A100 como no artigo)
- Implementar data loading paralelo
- Aplicar data augmentation durante treinamento
- Considerar mixed precision training para economia de memória

### Adaptações Necessárias:
- Ajustar para suas coordenadas geográficas específicas
- Modificar classes conforme necessidade do projeto
- Adaptar resolução de entrada conforme qualidade das imagens
- Implementar validação cruzada temporal para robustez