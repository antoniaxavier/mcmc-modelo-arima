# ARIMA usando MCMC
Ajuste bayesiano de um modelo AR(1) a uma série temporal diária, com estimação via MCMC (Gibbs/JAGS). Trabalho da disciplina de Inferência Bayesiana, feito em dupla com João Vitor Prisco.
Dados
`perfect night_ d1 _d184.txt` contém 184 observações diárias de streams da música Perfect Night, de 26/10/2023 a 27/04/2024. No script a série é dividida por 1000, então tudo (estimativas, previsões) está em milhares de streams.
Modelo

y_t = φ · y_{t-1} + ε_t, com ε_t ~ N(0, 1/τ)
Prioris:
Parâmetro	Priori	Papel
φ	Uniforme(−1, 1)	coeficiente autorregressivo, restrito à região de estacionariedade
τ	Gama(800, 1000)	precisão do erro
y₀	Normal(1391.599, variância 1.2e6)	valor inicial não observado, tratado como parâmetro
A estrutura hierárquica está em `hierarquia.png`. As deduções analíticas da verossimilhança e da condicional completa de y₀ estão nos PDFs correspondentes.
Estrutura
```
Script_Trabalho Bayesiana _ Antonia Xavier e João Vitor Prisco.R   # código
perfect night_ d1 _d184.txt                                        # dados
RelatórioBayesiana_Antonia_João.docx                               # relatório final
Função de Verossimilhança.pdf                                      # dedução da verossimilhança
Condicional completa y_0.pdf                                       # dedução da condicional de y0
hierarquia.png                                                     # diagrama do modelo
```
Como rodar
Requer JAGS instalado no sistema, além dos pacotes R:
```r
install.packages(c("R2jags", "coda", "ggplot2", "zoo"))
```
Com o diretório de trabalho apontando para a pasta do repositório, rode o script inteiro. Ele lê o `.txt`, monta a série com `zoo`, ajusta o modelo e gera os gráficos.
Configuração do MCMC
4 cadeias, 10.000 iterações por cadeia, burn-in de 2.000, thinning de 8 — resultando em 4.000 amostras da posteriori.
O que o script produz
Convergência: traceplots, diagnóstico de Geweke e Raftery–Lewis, ACF de cada parâmetro.
Inferência: médias e medianas a posteriori, intervalos HPD e intervalos de credibilidade por quantis (2,5% e 97,5%) para φ, τ e y₀; histogramas das posterioris marginais.
Previsão: um passo à frente (comparado com a observação real do dia 184) e simulação recursiva da distribuição preditiva, atualizando o último valor a cada iteração.
Autores
Antonia Xavier e João Vitor Prisco.
