# Detecção de Estresse Hídrico em Mudas de Açaí

Ferramenta baseada em Visão Computacional e Aprendizado de Máquina para detecção de estresse hídrico em mudas de açaí a partir de campanhas de filmagens.

## Estrutura do projeto

```
.
├── notebooks/      # Jupyter notebooks para experimentos e análises
├── old_scripts/    # Scripts antigos ou descontinuados
├── outputs/        # Resultados gerados pelos experimentos
├── rois/           # Regiões de interesse (ROIs) utilizadas nas análises
├── scripts/        # Scripts principais do projeto
├── videos/         # Vídeos utilizados nos experimentos
└── README.md
```

## Requisitos

- Python 3.12+
- ffmpeg (para processamento de vídeos)

### Bibliotecas

```bash
pip install \
    numpy \
    scipy \
    pandas \
    matplotlib \
    opencv-python \
    scikit-learn \
    scikit-image \
    jupyter \
    tqdm
```

Caso utilize PCA, ICA e processamento de sinais:

```bash
pip install \
    scipy \
    scikit-learn \
    pywavelets
```

## Organização dos vídeos

Os vídeos encontram-se em `videos/`.

Cada campanha contém:

```
videos/
└── camp_1
    └── AAAA-MM-DD/
            ├── 60/
            └── 240/
└── camp_2
    └── AAAA-MM-DD_1/
             ├── m/
    └── AAAA-MM-DD_2/
             ├── n/
└── camp_3
    └── AAAA-MM-DD_1/
             ├── m/
    └── AAAA-MM-DD_2/
             ├── n/
```

- Cada diretório da camp_1 contém quatro vídeos correspondentes a quatro ângulos da planta (rotação de 45° entre capturas).
- `60/`: vídeos gravados a 60 FPS.
- `240/`: vídeos gravados em câmera lenta.
- Os vídeos em **240 FPS** são armazenados com metadados de **25 FPS**, sendo necessária a correção do FPS durante o pré-processamento.

## Fluxo de processamento

1. Leitura dos vídeos.
2. Seleção da região de interesse (ROI).
3. Conversão para escala de cinza.
4. Redução da resolução.
5. Organização dos pixels em séries temporais.
6. PCA.
7. Blind Source Separation (CP-LA).
8. FFT/PSD.
9. Extração de frequências dominantes.

## Saídas

Os resultados são armazenados em `outputs/` e incluem:

- sinais processados;
- componentes principais;
- componentes independentes;
- espectros de frequência;
- figuras e gráficos.

## Licença

Projeto desenvolvido para pesquisa científica.