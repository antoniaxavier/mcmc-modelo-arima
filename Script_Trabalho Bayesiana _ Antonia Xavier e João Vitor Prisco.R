library(ggplot2)
library(zoo)
library("R2jags") 
library(coda)


dados <-  read.table("perfect night_ d1 _d184.txt")
data_inicio <- as.Date("2023-10-26")
data_fim <- as.Date("2024-04-27")


# Crie um vetor de datas
datas <- seq(from = data_inicio, to = data_fim, by = "day")
datas <- datas[-1]
# Crie a série temporal usando 'zoo'
streams <- zoo(dados, order.by = datas)
rm(datas)


#jags

itera = 10000  # número total de iterações
cad   = 4    # número de cadeias em paralelo que ele vai rodar 
burn  = itera/5 # burnin a ser retirado, somar o numero total planejado + burnin
thin  = 8    # espaçamento
n     = 184  # tamanho amostral

streams = dados$V1/1000

model_data = list(n = n, y = streams)
model_code = '
  model
  {
    # Priors
    phi ~ dunif(-1,1)
    tau ~ dgamma(800,1000)
    y0 ~ dnorm(1391.599, 1/1200000)
    
    
    # Likelihood
    for (i in 2:n) {
      y[i] ~ dnorm((mu[i]), tau) 
      mu[i] <-  phi * y[i - 1]
    }
  }'
model_parameters =  c("phi", "tau", "y0")


# Usando de fato a função jags:
model = jags(data = model_data, 
             parameters.to.save = model_parameters, 
             model.file = textConnection(model_code), 
             n.chains = cad,  
             n.iter = itera,  
             n.burnin = burn, 
             n.thin = thin,   
             DIC = FALSE)
model

#Analise de convergencia das cadeias das cadeias do MCMC.


traceplot(model) 
plot(model)
jags = mcmc(model$BUGSoutput$sims.matrix) #transformando o resultado final em um objeto mcmc pra usar o pacote coda.

geweke.diag(jags)
geweke.plot(jags)

raftery.diag(jags)


phi_chapeu <- jags[,1]
tau_chapeu <-  jags[,2]
y0_chapeu <- jags[,3]

par(mfrow= c(1,3))
plot(as.vector(phi_chapeu), type = "l", main = "Valores de phi por iteração")
abline(h=mean(phi_chapeu),col=2,lwd=2)
plot(as.vector(tau_chapeu), type = "l", main = "Valores de tau por iteração")
abline(h=mean(tau_chapeu),col=2,lwd=2)
plot(as.vector(y0_chapeu),type="l",  main = "Valores de y0 por iteração")
abline(h=mean(y0_chapeu),col=2,lwd=2)


#(e) Calcule estimativas pontuais e intervalares a posteriori para os parˆametros do modelo e comente.
median(tau_chapeu)    
HPDinterval(tau_chapeu)

median(phi_chapeu)    
HPDinterval(phi_chapeu)

median(y0_chapeu)    
HPDinterval(y0_chapeu)

mean(phi_chapeu)

quantile(phi_chapeu,0.975)
quantile(phi_chapeu,0.025)

quantile(tau_chapeu,0.975)
quantile(tau_chapeu,0.025)

quantile(y0_chapeu,0.975)
quantile(y0_chapeu,0.025)

# Distribuições a posteriori
par(mfrow=c(1,3))
hist(phi_chapeu,prob=T,ylab="",xlab="",main=expression(phi))
points(mean(phi_chapeu),0,col=3,lwd=4)
points(HPDinterval(phi_chapeu)
       [1],0,col=3,lwd=4)
points(HPDinterval(phi_chapeu)
       [2],0,col=3,lwd=4)
hist(tau_chapeu,prob=T,ylab="",xlab="",main=expression(tau^{-1}))
points(mean(tau_chapeu),0,col=3,lwd=4)
points(HPDinterval(tau_chapeu)
       [1],0,col=3,lwd=4)
points(HPDinterval(tau_chapeu)
       [2],0,col=3,lwd=4)
hist(y0_chapeu,prob=T,ylab="",xlab="",main=expression(y[0]))
points(mean(y0_chapeu),0,col=3,lwd=4)
points(HPDinterval(y0_chapeu)
       [1],0,col=3,lwd=4)
points(HPDinterval(y0_chapeu)
       [2],0,col=3,lwd=4)


acf(phi_chapeu)
acf(tau_chapeu)
acf(y0_chapeu)


#questao f
set.seed(123)
#1 passo a frente
y_pred <- rnorm(1,phi_chapeu[1]*streams[183],(1/tau_chapeu[1]))
y_pred
streams[184]
# Último valor observado
y_last <- streams[length(streams)]

# Loop para gerar as previsões n passos à frente
for (i in 1:4000) {
  # Calcular a média preditiva
  mean_pred <- phi_chapeu * y_last
  
  # Calcular o desvio padrão preditivo
  sd_pred <- sqrt(1 / tau_chapeu)
  
  # Gerar um valor preditivo a partir da distribuição normal
  y_pred[i] <- rnorm(1, mean_pred, sd_pred)
  
  # Atualizar o último valor para a próxima iteração
  y_last <- y_pred[i]
}
par(mfrow= c(1,1))
hist(y_pred)
mean(y_pred)
