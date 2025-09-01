# OCT Image Analysis - Treinamento e Análise Completa

## Introdução

Este projeto implementa um pipeline completo de treinamento e análise de modelos de deep learning para classificação de imagens de Tomografia de Coerência Óptica (OCT). O sistema é capaz de treinar redes neurais convolucionais customizadas e analisar o desempenho através de técnicas como Grad-CAM, matriz de confusão e curvas ROC.

### Funcionalidades

- **Treinamento de modelo customizado** com arquitetura baseada em separable convolutions
- **Data augmentation** para melhorar a generalização
- **Class weights balanceados** com boost para classes minoritárias
- **Análise de desempenho** com métricas detalhadas
- **Visualização Grad-CAM** para interpretabilidade
- **Suporte a GPU** (CUDA) para treinamento acelerado

### Classes de Diagnóstico

- **CNV** - Neovascularização Coroidal (26.218 amostras de treino)
- **DME** - Edema Macular Diabético (8.118 amostras de treino) 
- **DRUSEN** - Drusas (6.206 amostras de treino)
- **NORMAL** - Retina Normal (35.973 amostras de treino)

## Pré-requisitos

### Sistema Operacional
Este projeto foi desenvolvido e testado no **WSL2 (Ubuntu)** com suporte a GPU NVIDIA. É recomendado usar:
- **Ubuntu 20.04+** (via WSL2 no Windows ou nativo)
- **NVIDIA GPU** com drivers CUDA instalados
- **CUDA 12.x** e **cuDNN** configurados

### Software Necessário
- **Python 3.8-3.11**
- **Conda/Miniconda** para gerenciamento de ambientes
- **Jupyter Notebook** ou **JupyterLab**

## Configuração do Ambiente

### 1. Criação do Ambiente Conda

```bash
# Criar ambiente com Python 3.11
conda create -n tf_gpu python=3.11 -y

# Ativar o ambiente
conda activate tf_gpu
```

### 2. Instalação das Dependências

```bash
# Instalar dependências via pip
pip install -r requirements.txt

# OU instalar manualmente as principais:
pip install tensorflow==2.19.0
pip install keras==3.10.0
pip install jupyter
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install opencv-python
pip install pillow
```

### 3. Verificação da Instalação GPU

Execute este código Python para verificar se o TensorFlow detecta sua GPU:

```python
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
```

## Estrutura do Projeto

```
projeto/
│
├── notebook_analysis.ipynb    # Notebook principal de treinamento
├── requirements.txt          # Dependências do projeto
├── environment.yml          # Arquivo de ambiente Conda (opcional)
├── README.md               # Este arquivo
│
├── data/                   # Dados do projeto
│   └── Dataset - train+val+test/
│       ├── train/
│       ├── val/
│       └── test/
│
└── modelos baixados/       # Modelos salvos
    ├── best_model_custom.keras
    └── history.pkl
```

## Configuração dos Dados

### 1. Dataset OCT

Organize seu dataset na seguinte estrutura:

```
/caminho/para/Dataset - train+val+test/
├── train/
│   ├── CNV/
│   ├── DME/
│   ├── DRUSEN/
│   └── NORMAL/
├── val/
│   ├── CNV/
│   ├── DME/
│   ├── DRUSEN/
│   └── NORMAL/
└── test/
    ├── CNV/
    ├── DME/
    ├── DRUSEN/
    └── NORMAL/
```

### 2. Configuração de Caminhos

No notebook, ajuste as seguintes variáveis para seu sistema:

```python
DATA_DIR = "/mnt/d/Dataset - train+val+test"  # Caminho para o dataset
MODEL_SAVE_DIR = "/mnt/d/modelos baixados"    # Onde salvar os modelos
```

**Para usuários Windows com WSL:**
- Dados no drive D: → `/mnt/d/caminho/para/dados`
- Dados no drive C: → `/mnt/c/caminho/para/dados`

## Executando o Projeto

### 1. Iniciar Jupyter Notebook

```bash
# Ativar o ambiente
conda activate tf_gpu

# Iniciar Jupyter
jupyter notebook
# OU para JupyterLab
jupyter lab
```

### 2. Abrir o Notebook

1. Navegue até o arquivo `test-oct-train (code final).ipynb`
2. Execute as células sequencialmente
3. Ajuste os parâmetros conforme necessário

### 3. Etapas do Workflow

#### **Fase 1: Preparação dos Dados**
- Carregamento dos datasets (train/val/test)
- Aplicação de data augmentation
- Configuração de class weights

#### **Fase 2: Treinamento (Opcional)**
- Descomente o código de treinamento se necessário
- O modelo já treinado está disponível em `best_model_custom.keras`

#### **Fase 3: Avaliação**
- Carregamento do modelo pré-treinado
- Avaliação no dataset de teste
- Geração de métricas de desempenho

#### **Fase 4: Análise Visual**
- Matriz de confusão
- Curvas ROC multi-classe
- Visualização do histórico de treinamento
- Análise Grad-CAM para interpretabilidade

## Parâmetros Principais

### Configurações do Modelo
```python
IMG_SIZE = (224, 224)        # Tamanho das imagens
BATCH_SIZE = 64              # Tamanho do batch
NUM_CLASSES = 4              # Número de classes
LAST_CONV_LAYER_NAME = "separable_conv2d_13"  # Camada para Grad-CAM
```

### Hiperparâmetros de Treinamento
- **Épocas:** 50 (com early stopping)
- **Otimizador:** Adam
- **Loss:** Categorical Crossentropy
- **Callbacks:** EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

## Resultados Esperados

### Performance do Modelo
- **Acurácia no teste:** ~95.94%
- **Loss no teste:** ~0.1273

### Métricas por Classe
O modelo apresenta alta precisão para todas as classes, com particular atenção às classes minoritárias (DME e DRUSEN) através do uso de class weights balanceados.

## Solução de Problemas

### Erro de GPU/CUDA
```bash
# Verificar instalação CUDA
nvcc --version
nvidia-smi

# Reinstalar TensorFlow com suporte GPU
pip uninstall tensorflow
pip install tensorflow==2.19.0
```

### Erro de Memória
```python
# Reduzir batch size
BATCH_SIZE = 32  # ou menor

# Usar mixed precision
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")
```

### Problema com Caminhos (WSL)
```bash
# Verificar montagem dos drives
ls /mnt/c/
ls /mnt/d/

# Ajustar permissões se necessário
sudo chmod -R 755 /mnt/d/seu_dataset/
```

## Arquivos de Saída

Após a execução, o projeto gera:

- `best_model_custom.keras` - Modelo treinado
- `history.pkl` - Histórico do treinamento
- Gráficos de análise (matplotlib)
- Visualizações Grad-CAM

## Extensões e Melhorias

### Próximos Passos
1. **Transfer Learning** com modelos pré-treinados (ResNet, EfficientNet)
2. **Ensemble Methods** para melhorar performance
3. **Cross-validation** para validação mais robusta
4. **Deployment** com Flask/FastAPI (já implementado em `app.py`)

### Otimizações
- **TensorRT** para inferência acelerada
- **Quantização** do modelo para produção
- **Distributed training** para datasets maiores

## Contribuição

Para contribuir:

1. Fork o repositório
2. Crie uma branch: `git checkout -b feature/nova-feature`
3. Commit: `git commit -am 'Adiciona nova feature'`
4. Push: `git push origin feature/nova-feature`
5. Abra um Pull Request

## Suporte

### Problemas Comuns
- **Verificar versões:** TensorFlow 2.19.0, Keras 3.10.0
- **Verificar caminhos:** Ajustar DATA_DIR e MODEL_SAVE_DIR
- **Verificar GPU:** Usar `nvidia-smi` para monitoramento

### Recursos Úteis
- [TensorFlow GPU Support](https://www.tensorflow.org/install/gpu)
- [WSL2 CUDA Setup](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [Keras Documentation](https://keras.io/)

---

**Nota:** Este é um projeto de pesquisa para análise médica. Os resultados devem sempre ser validados por profissionais qualificados antes de uso clínico.
